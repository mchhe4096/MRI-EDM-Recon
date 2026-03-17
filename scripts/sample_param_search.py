import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.datasets import PtKspaceReconDataset
from src.diffusion.edm import EDMTrainer
from src.models.unet_v2 import UNetV2


def parse_list(text: str, cast):
    vals = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(cast(p))
    return vals


def parse_bool_list(text: str):
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}
    vals = []
    for p in text.split(","):
        p = p.strip().lower()
        if not p:
            continue
        if p in truthy:
            vals.append(True)
        elif p in falsy:
            vals.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {p}")
    return vals


def parse_str_list(text: str):
    vals = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(p)
    return vals


def to_mag(x2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(x2[0] * x2[0] + x2[1] * x2[1] + eps)


def psnr_mag(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((pred - gt) ** 2)
    maxv = torch.max(gt).clamp_min(eps)
    return float((20 * torch.log10(maxv / torch.sqrt(mse + eps))).item())


def nmse_mag(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.sum((pred - gt) ** 2)
    den = torch.sum(gt ** 2).clamp_min(eps)
    return float((num / den).item())


def gaussian_window(window_size: int = 11, sigma: float = 1.5, device: torch.device | None = None):
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    w2d = torch.outer(g, g)
    return w2d.view(1, 1, window_size, window_size)


def ssim_mag(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> float:
    pred4 = pred.view(1, 1, pred.shape[0], pred.shape[1])
    gt4 = gt.view(1, 1, gt.shape[0], gt.shape[1])
    win = gaussian_window(window_size=window_size, sigma=sigma, device=pred.device)

    mu_x = F.conv2d(pred4, win, padding=window_size // 2)
    mu_y = F.conv2d(gt4, win, padding=window_size // 2)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred4 * pred4, win, padding=window_size // 2) - mu_x2
    sigma_y2 = F.conv2d(gt4 * gt4, win, padding=window_size // 2) - mu_y2
    sigma_xy = F.conv2d(pred4 * gt4, win, padding=window_size // 2) - mu_xy

    data_range = (torch.max(gt) - torch.min(gt)).clamp_min(eps)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + eps)
    return float(ssim_map.mean().item())


def log_kernel(size: int = 15, sigma: float = 1.5, device: torch.device | None = None):
    if size % 2 == 0:
        raise ValueError("LoG kernel size must be odd.")
    r = (size - 1) // 2
    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1, dtype=torch.float32, device=device),
        torch.arange(-r, r + 1, dtype=torch.float32, device=device),
        indexing="ij",
    )
    rr = xs * xs + ys * ys
    sigma2 = sigma * sigma
    kernel = ((rr - 2.0 * sigma2) / (sigma2 * sigma2)) * torch.exp(-rr / (2.0 * sigma2))
    kernel = kernel - kernel.mean()
    return kernel.view(1, 1, size, size)


def hfen_mag(pred: torch.Tensor, gt: torch.Tensor, sigma: float = 1.5, size: int = 15, eps: float = 1e-12) -> float:
    pred4 = pred.view(1, 1, pred.shape[0], pred.shape[1])
    gt4 = gt.view(1, 1, gt.shape[0], gt.shape[1])
    k = log_kernel(size=size, sigma=sigma, device=pred.device)
    pred_f = F.conv2d(pred4, k, padding=size // 2)
    gt_f = F.conv2d(gt4, k, padding=size // 2)
    num = torch.norm(pred_f - gt_f, p=2)
    den = torch.norm(gt_f, p=2).clamp_min(eps)
    return float((num / den).item())


def load_lpips(lpips_net: str, device: torch.device, skip_if_unavailable: bool):
    try:
        import lpips  # type: ignore
    except Exception:
        if skip_if_unavailable:
            print("[warn] lpips package is not installed; LPIPS will be NaN.")
            return None
        raise RuntimeError(
            "lpips package is required for LPIPS metric.\n"
            "Install with: pip install lpips"
        )
    model = lpips.LPIPS(net=lpips_net).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def lpips_mag(pred: torch.Tensor, gt: torch.Tensor, model, device: torch.device, eps: float = 1e-12) -> float:
    if model is None:
        return float("nan")
    scale = torch.max(gt).clamp_min(eps)
    pred_n = (pred / scale).clamp(0.0, 1.0)
    gt_n = (gt / scale).clamp(0.0, 1.0)
    pred_t = pred_n.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    gt_t = gt_n.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    pred_t = pred_t * 2.0 - 1.0
    gt_t = gt_t * 2.0 - 1.0
    with torch.no_grad():
        val = model(pred_t, gt_t)
    return float(val.item())


def summarize(vals: list[float]):
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def sanitize_for_json(obj: Any):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def build_configs(args):
    steps_list = parse_list(args.steps_list, int)
    dc_start_list = parse_list(args.dc_start_list, float)
    dc_every_list = parse_list(args.dc_every_list, int)
    dc_lam_list = parse_list(args.dc_lam_list, float)
    dc_ramp_list = parse_bool_list(args.dc_ramp_list)
    init_mode_list = [m.lower() for m in parse_str_list(args.init_mode_list)]
    init_blend_list = parse_list(args.init_blend_list, float)

    valid_init_modes = {"noise", "zf", "blend"}
    for mode in init_mode_list:
        if mode not in valid_init_modes:
            raise ValueError(f"Invalid init_mode: {mode}. Expected one of {sorted(valid_init_modes)}")

    init_cfgs: list[tuple[str, float | None]] = []
    for mode in init_mode_list:
        if mode == "blend":
            if len(init_blend_list) == 0:
                raise ValueError("init_mode_list includes 'blend' but init_blend_list is empty.")
            for b in init_blend_list:
                init_cfgs.append(("blend", float(b)))
        else:
            init_cfgs.append((mode, None))

    configs: list[dict[str, Any]] = []
    for steps in steps_list:
        for init_mode, init_blend in init_cfgs:
            if args.include_no_dc_baseline:
                configs.append(
                    {
                        "steps": steps,
                        "dc": False,
                        "dc_start": None,
                        "dc_every": None,
                        "dc_lam": None,
                        "dc_ramp": None,
                        "init_mode": init_mode,
                        "init_blend": init_blend,
                    }
                )
            for dc_start, dc_every, dc_lam, dc_ramp in itertools.product(
                dc_start_list, dc_every_list, dc_lam_list, dc_ramp_list
            ):
                configs.append(
                    {
                        "steps": steps,
                        "dc": True,
                        "dc_start": float(dc_start),
                        "dc_every": int(dc_every),
                        "dc_lam": float(dc_lam),
                        "dc_ramp": bool(dc_ramp),
                        "init_mode": init_mode,
                        "init_blend": init_blend,
                    }
                )
    return configs


def parse_args():
    p = argparse.ArgumentParser(description="Grid search sampling params and evaluate MRI metrics.")
    p.add_argument("--ckpt", type=str, default="../outputs/ckpts/last.pt")
    p.add_argument("--val_root", type=str, default="/root/autodl-tmp/val_v2")

    p.add_argument("--accel", type=int, default=None, help="Override accel from checkpoint args.")
    p.add_argument("--center_frac", type=float, default=None, help="Override center_frac from checkpoint args.")
    p.add_argument(
        "--include_mask_channel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override include_mask_channel from checkpoint args.",
    )
    p.add_argument("--base_ch", type=int, default=None, help="Override UNet base channels from checkpoint args.")
    p.add_argument("--sigma_data", type=float, default=None, help="Override sigma_data from checkpoint args.")

    p.add_argument("--num_cases", type=int, default=0, help="0 means all validation cases.")
    p.add_argument("--num_samples", type=int, default=1, help="Samples per case per config.")
    p.add_argument(
        "--aggregate",
        type=str,
        default="complex_mean",
        choices=["complex_mean", "magnitude_mean"],
        help="How to aggregate multi-sample outputs before metrics.",
    )
    p.add_argument("--seed", type=int, default=1234, help="Base seed for deterministic sampling.")
    p.add_argument(
        "--init_mode_list",
        type=str,
        default="zf",
        help="Comma-separated sampler init modes: noise,zf,blend",
    )
    p.add_argument(
        "--init_blend_list",
        type=str,
        default="0.5",
        help="Comma-separated blend alpha(s) for init_mode=blend (alpha*x_zf + (1-alpha)*noise).",
    )

    p.add_argument("--steps_list", type=str, default="40")
    p.add_argument("--dc_start_list", type=str, default="0.6,0.8")
    p.add_argument("--dc_every_list", type=str, default="2,3")
    p.add_argument("--dc_lam_list", type=str, default="0.08,0.12,0.15")
    p.add_argument("--dc_ramp_list", type=str, default="0,1")
    p.add_argument(
        "--include_no_dc_baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate no-DC baseline for each steps setting.",
    )

    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument(
        "--skip_lpips_if_unavailable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If lpips package is missing, fill LPIPS with NaN instead of raising.",
    )

    p.add_argument("--hfen_sigma", type=float, default=1.5)
    p.add_argument("--hfen_kernel", type=int, default=15)
    p.add_argument("--rank_metric", type=str, default="psnr", choices=["psnr", "ssim", "nmse", "lpips", "hfen"])
    p.add_argument("--topk", type=int, default=10)

    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--outdir", type=str, default="../outputs/param_search")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})

    accel = int(train_args.get("accel", 4) if args.accel is None else args.accel)
    center_frac = float(train_args.get("center_frac", 0.08) if args.center_frac is None else args.center_frac)
    include_mask_channel = bool(
        train_args.get("include_mask_channel", True)
        if args.include_mask_channel is None
        else args.include_mask_channel
    )
    base_ch = int(train_args.get("base_ch", 128) if args.base_ch is None else args.base_ch)
    sigma_min = float(train_args.get("sigma_min", 0.002))
    sigma_max = float(train_args.get("sigma_max", 80.0))
    sigma_data = float(train_args.get("sigma_data", 0.5) if args.sigma_data is None else args.sigma_data)

    cond_ch = 3 if include_mask_channel else 2
    model = UNetV2(x_ch=2, cond_ch=cond_ch, out_ch=2, base_ch=base_ch).to(device)
    if args.use_ema:
        if "ema_model" not in ckpt:
            raise RuntimeError("Checkpoint has no ema_model; run with --no-use_ema to use raw model weights.")
        weights = ckpt["ema_model"]
        weights_name = "ema_model"
    else:
        weights = ckpt["model"]
        weights_name = "model"
    model.load_state_dict(weights, strict=True)
    model.eval()

    trainer = EDMTrainer(sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data)
    lpips_model = load_lpips(args.lpips_net, device=device, skip_if_unavailable=args.skip_lpips_if_unavailable)

    ds = PtKspaceReconDataset(
        root=args.val_root,
        accel=accel,
        center_frac=center_frac,
        include_mask_channel=include_mask_channel,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    total_cases = len(ds) if args.num_cases == 0 else min(len(ds), int(args.num_cases))

    case_cache: list[dict[str, Any]] = []
    for i, batch in enumerate(loader):
        if i >= total_cases:
            break
        path_item = batch.get("path", "")
        if isinstance(path_item, (list, tuple)):
            path_item = path_item[0] if len(path_item) > 0 else ""
        case_cache.append(
            {
                "case_idx": i,
                "path": str(path_item),
                "target": batch["target"],
                "cond": batch["cond"],
                "k_us": batch["k_us"],
                "mask": batch["mask"],
            }
        )

    configs = build_configs(args)
    print(
        f"[info] device={device} cases={len(case_cache)} num_samples={args.num_samples} "
        f"configs={len(configs)} weights={weights_name}"
    )

    zf_psnrs: list[float] = []
    zf_ssims: list[float] = []
    zf_nmses: list[float] = []
    zf_lpipss: list[float] = []
    zf_hfens: list[float] = []
    zf_per_case_rows: list[dict[str, Any]] = []
    for c in case_cache:
        gt2 = c["target"][0]
        zf2 = c["cond"][0, :2]
        zf_mag = to_mag(zf2)
        gt_mag = to_mag(gt2)
        psnr_v = psnr_mag(zf_mag, gt_mag)
        ssim_v = ssim_mag(zf_mag, gt_mag)
        nmse_v = nmse_mag(zf_mag, gt_mag)
        lpips_v = lpips_mag(zf_mag, gt_mag, model=lpips_model, device=device)
        hfen_v = hfen_mag(zf_mag, gt_mag, sigma=float(args.hfen_sigma), size=int(args.hfen_kernel))
        zf_psnrs.append(psnr_v)
        zf_ssims.append(ssim_v)
        zf_nmses.append(nmse_v)
        zf_lpipss.append(lpips_v)
        zf_hfens.append(hfen_v)
        zf_per_case_rows.append(
            {
                "case_idx": c["case_idx"],
                "path": c["path"],
                "psnr": psnr_v,
                "ssim": ssim_v,
                "nmse": nmse_v,
                "lpips": lpips_v,
                "hfen": hfen_v,
            }
        )

    zf_psnr_mean, zf_psnr_std = summarize(zf_psnrs)
    zf_ssim_mean, zf_ssim_std = summarize(zf_ssims)
    zf_nmse_mean, zf_nmse_std = summarize(zf_nmses)
    zf_lpips_mean, zf_lpips_std = summarize(zf_lpipss)
    zf_hfen_mean, zf_hfen_std = summarize(zf_hfens)
    zf_baseline = {
        "name": "zf",
        "num_cases": len(case_cache),
        "psnr_mean": zf_psnr_mean,
        "psnr_std": zf_psnr_std,
        "ssim_mean": zf_ssim_mean,
        "ssim_std": zf_ssim_std,
        "nmse_mean": zf_nmse_mean,
        "nmse_std": zf_nmse_std,
        "lpips_mean": zf_lpips_mean,
        "lpips_std": zf_lpips_std,
        "hfen_mean": zf_hfen_mean,
        "hfen_std": zf_hfen_std,
    }
    print(
        "[baseline zf] "
        f"PSNR={zf_psnr_mean:.4f} SSIM={zf_ssim_mean:.4f} NMSE={zf_nmse_mean:.6f} "
        f"LPIPS={zf_lpips_mean:.6f} HFEN={zf_hfen_mean:.6f}"
    )

    summary_rows: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for cfg_id, cfg in enumerate(configs):
        psnrs: list[float] = []
        ssims: list[float] = []
        nmses: list[float] = []
        lpipss: list[float] = []
        hfens: list[float] = []

        for c in case_cache:
            target = c["target"].to(device)
            cond = c["cond"].to(device)
            k_us = c["k_us"].to(device)
            mask = c["mask"].to(device)

            preds = []
            for n in range(int(args.num_samples)):
                seed = int(args.seed + cfg_id * 1_000_000 + c["case_idx"] * 10_000 + n)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

                pred = trainer.sample(
                    model=model,
                    shape=target.shape,
                    cond=cond,
                    steps=int(cfg["steps"]),
                    dc=bool(cfg["dc"]),
                    k_us=k_us,
                    mask=mask,
                    dc_start=0.6 if cfg["dc_start"] is None else float(cfg["dc_start"]),
                    dc_every=2 if cfg["dc_every"] is None else int(cfg["dc_every"]),
                    dc_lam=0.15 if cfg["dc_lam"] is None else float(cfg["dc_lam"]),
                    dc_ramp=False if cfg["dc_ramp"] is None else bool(cfg["dc_ramp"]),
                    init_mode=str(cfg["init_mode"]),
                    init_blend=0.5 if cfg["init_blend"] is None else float(cfg["init_blend"]),
                )
                preds.append(pred[0].detach().cpu())

            stack = torch.stack(preds, dim=0)  # [N,2,H,W]
            gt2 = target[0].detach().cpu()

            if args.aggregate == "complex_mean":
                pred2 = stack.mean(dim=0)
                pred_mag = to_mag(pred2)
            else:
                pred_mag = torch.sqrt(stack[:, 0] ** 2 + stack[:, 1] ** 2 + 1e-12).mean(dim=0)

            gt_mag = to_mag(gt2)
            psnr_v = psnr_mag(pred_mag, gt_mag)
            ssim_v = ssim_mag(pred_mag, gt_mag)
            nmse_v = nmse_mag(pred_mag, gt_mag)
            lpips_v = lpips_mag(pred_mag, gt_mag, model=lpips_model, device=device)
            hfen_v = hfen_mag(pred_mag, gt_mag, sigma=float(args.hfen_sigma), size=int(args.hfen_kernel))

            psnrs.append(psnr_v)
            ssims.append(ssim_v)
            nmses.append(nmse_v)
            lpipss.append(lpips_v)
            hfens.append(hfen_v)

            per_case_rows.append(
                {
                    "config_id": cfg_id,
                    "init_mode": cfg["init_mode"],
                    "init_blend": cfg["init_blend"],
                    "case_idx": c["case_idx"],
                    "path": c["path"],
                    "psnr": psnr_v,
                    "ssim": ssim_v,
                    "nmse": nmse_v,
                    "lpips": lpips_v,
                    "hfen": hfen_v,
                }
            )

        psnr_mean, psnr_std = summarize(psnrs)
        ssim_mean, ssim_std = summarize(ssims)
        nmse_mean, nmse_std = summarize(nmses)
        lpips_mean, lpips_std = summarize(lpipss)
        hfen_mean, hfen_std = summarize(hfens)

        row = {
            "config_id": cfg_id,
            "init_mode": cfg["init_mode"],
            "init_blend": cfg["init_blend"],
            "steps": cfg["steps"],
            "dc": cfg["dc"],
            "dc_start": cfg["dc_start"],
            "dc_every": cfg["dc_every"],
            "dc_lam": cfg["dc_lam"],
            "dc_ramp": cfg["dc_ramp"],
            "num_cases": len(case_cache),
            "num_samples": int(args.num_samples),
            "aggregate": args.aggregate,
            "psnr_mean": psnr_mean,
            "psnr_std": psnr_std,
            "ssim_mean": ssim_mean,
            "ssim_std": ssim_std,
            "nmse_mean": nmse_mean,
            "nmse_std": nmse_std,
            "lpips_mean": lpips_mean,
            "lpips_std": lpips_std,
            "hfen_mean": hfen_mean,
            "hfen_std": hfen_std,
        }
        summary_rows.append(row)
        print(
            f"[cfg {cfg_id:03d}] init={cfg['init_mode']}"
            + (f"(a={cfg['init_blend']:.2f})" if cfg["init_blend"] is not None else "")
            + f" steps={cfg['steps']} dc={cfg['dc']} "
            f"start={cfg['dc_start']} every={cfg['dc_every']} lam={cfg['dc_lam']} ramp={cfg['dc_ramp']}  "
            f"PSNR={psnr_mean:.4f} SSIM={ssim_mean:.4f} NMSE={nmse_mean:.6f} "
            f"LPIPS={lpips_mean:.6f} HFEN={hfen_mean:.6f}"
        )

    higher_better = {"psnr", "ssim"}
    rank_key = f"{args.rank_metric}_mean"
    reverse = args.rank_metric in higher_better
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: (-math.inf if np.isnan(r[rank_key]) else r[rank_key]),
        reverse=reverse,
    )
    if not reverse:
        summary_rows_sorted = sorted(
            summary_rows,
            key=lambda r: (math.inf if np.isnan(r[rank_key]) else r[rank_key]),
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "summary.csv"
    per_case_path = outdir / "per_case.csv"
    zf_baseline_path = outdir / "baseline_zf.json"
    zf_per_case_path = outdir / "baseline_zf_per_case.csv"
    topk_path = outdir / "topk.json"
    meta_path = outdir / "meta.json"

    if len(summary_rows) > 0:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    if len(per_case_rows) > 0:
        with per_case_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_case_rows)
    if len(zf_per_case_rows) > 0:
        with zf_per_case_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(zf_per_case_rows[0].keys()))
            writer.writeheader()
            writer.writerows(zf_per_case_rows)
    with zf_baseline_path.open("w") as f:
        json.dump(sanitize_for_json(zf_baseline), f, indent=2)

    topk = summary_rows_sorted[: max(1, int(args.topk))]
    with topk_path.open("w") as f:
        json.dump(sanitize_for_json(topk), f, indent=2)

    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": time.time() - t_start,
        "ckpt": args.ckpt,
        "weights": weights_name,
        "val_root": args.val_root,
        "device": str(device),
        "num_cases": len(case_cache),
        "num_samples": int(args.num_samples),
        "aggregate": args.aggregate,
        "rank_metric": args.rank_metric,
        "lpips_available": lpips_model is not None,
        "zf_baseline": zf_baseline,
        "search_space": {
            "steps_list": args.steps_list,
            "dc_start_list": args.dc_start_list,
            "dc_every_list": args.dc_every_list,
            "dc_lam_list": args.dc_lam_list,
            "dc_ramp_list": args.dc_ramp_list,
            "include_no_dc_baseline": bool(args.include_no_dc_baseline),
            "init_mode_list": args.init_mode_list,
            "init_blend_list": args.init_blend_list,
        },
    }
    with meta_path.open("w") as f:
        json.dump(sanitize_for_json(meta), f, indent=2)

    print(f"[done] wrote: {summary_path}")
    print(f"[done] wrote: {per_case_path}")
    print(f"[done] wrote: {zf_per_case_path}")
    print(f"[done] wrote: {zf_baseline_path}")
    print(f"[done] wrote: {topk_path}")
    print(f"[done] wrote: {meta_path}")
    if len(topk) > 0:
        best = topk[0]
        print(
            f"[best] cfg={best['config_id']} {args.rank_metric}={best[rank_key]:.6f} "
            f"init={best['init_mode']}"
            + (f"(a={best['init_blend']:.2f})" if best["init_blend"] is not None else "")
            + " "
            f"steps={best['steps']} dc={best['dc']} start={best['dc_start']} "
            f"every={best['dc_every']} lam={best['dc_lam']} ramp={best['dc_ramp']}"
        )


if __name__ == "__main__":
    main()
