import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
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


def normalize_complex_np(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mag_std = float(np.abs(data).std())
    return data / (mag_std + eps)


def zscore_np(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (data - float(data.mean())) / (float(data.std()) + eps)


def author_prepare_mag_from_complex2(x2: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    real = x2[0].detach().cpu().numpy().astype(np.float32)
    imag = x2[1].detach().cpu().numpy().astype(np.float32)
    cimg = real + 1j * imag
    cimg = normalize_complex_np(cimg, eps=eps)
    mag = np.abs(cimg).astype(np.float32)
    return zscore_np(mag, eps=eps).astype(np.float32)


def author_prepare_mag_from_mag(mag: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    arr = mag.detach().cpu().numpy().astype(np.float32)
    return zscore_np(arr, eps=eps).astype(np.float32)


def parse_volume_and_slice(path_str: str, fallback_idx: int) -> tuple[str, int]:
    if not path_str:
        return "unknown", int(fallback_idx)
    stem = Path(path_str).stem
    m = re.match(r"(.+?)_(\d+)$", stem)
    if m is not None:
        return m.group(1), int(m.group(2))
    return stem, int(fallback_idx)


def psnr_from_arrays(pred: np.ndarray, gt: np.ndarray, data_range: float, eps: float = 1e-12) -> float:
    if not np.isfinite(data_range) or data_range <= eps:
        return float("nan")
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if not np.isfinite(mse):
        return float("nan")
    return float(20.0 * np.log10((data_range + eps) / np.sqrt(mse + eps)))


def author_volume_psnr(
    pred_stack_shw: np.ndarray,
    gt_stack_shw: np.ndarray,
    crop_head: int,
    crop_tail: int,
    range_mode: str,
    eps: float,
) -> tuple[float, int]:
    if pred_stack_shw.ndim != 3 or gt_stack_shw.ndim != 3:
        raise ValueError("expected stack arrays with shape [S,H,W]")

    s_pred = int(pred_stack_shw.shape[0])
    s_gt = int(gt_stack_shw.shape[0])
    s = min(s_pred, s_gt)
    if s <= 0:
        return float("nan"), 0

    pred = pred_stack_shw[:s]
    gt = gt_stack_shw[:s]

    start = max(0, int(crop_head))
    end = s - max(0, int(crop_tail))
    if end <= start:
        return float("nan"), 0

    pred = pred[start:end]
    gt = gt[start:end]
    if pred.shape[0] == 0:
        return float("nan"), 0

    if range_mode == "volume":
        dr = float(np.max(gt) - np.min(gt))
        return psnr_from_arrays(pred, gt, dr, eps=eps), int(pred.shape[0])

    vals = []
    for i in range(pred.shape[0]):
        dr_i = float(np.max(gt[i]) - np.min(gt[i]))
        vals.append(psnr_from_arrays(pred[i], gt[i], dr_i, eps=eps))
    finite = np.asarray(vals, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), int(pred.shape[0])
    return float(finite.mean()), int(pred.shape[0])


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
    p = argparse.ArgumentParser(
        description=(
            "Grid search sampling params and evaluate PSNR with author-aligned protocol. "
            "This script is a comparison-only copy and does not modify original pipeline files."
        )
    )
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

    p.add_argument(
        "--metric_protocol",
        type=str,
        default="both",
        choices=["author", "current", "both"],
        help="Author-aligned metric only, current metric only, or both.",
    )
    p.add_argument(
        "--rank_metric",
        type=str,
        default="author_psnr",
        choices=["author_psnr", "current_psnr"],
        help="Metric used to sort top-k configs.",
    )
    p.add_argument(
        "--author_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply author-style normalize_complex + zscore before PSNR.",
    )
    p.add_argument(
        "--author_crop_head",
        type=int,
        default=4,
        help="Author-style volume crop start index (equivalent to [...,4:-1]).",
    )
    p.add_argument(
        "--author_crop_tail",
        type=int,
        default=1,
        help="Author-style volume crop tail count (equivalent to [...,4:-1]).",
    )
    p.add_argument(
        "--author_range",
        type=str,
        default="volume",
        choices=["volume", "slice"],
        help="PSNR data_range mode under author protocol.",
    )
    p.add_argument("--author_eps", type=float, default=1e-8)

    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--outdir", type=str, default="../outputs/param_search_comparison")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.metric_protocol == "author" and args.rank_metric != "author_psnr":
        raise ValueError("metric_protocol=author requires rank_metric=author_psnr")
    if args.metric_protocol == "current" and args.rank_metric != "current_psnr":
        raise ValueError("metric_protocol=current requires rank_metric=current_psnr")

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
        volume_id, slice_idx = parse_volume_and_slice(str(path_item), fallback_idx=i)
        case_cache.append(
            {
                "case_idx": i,
                "path": str(path_item),
                "volume_id": volume_id,
                "slice_idx": int(slice_idx),
                "target": batch["target"],
                "cond": batch["cond"],
                "k_us": batch["k_us"],
                "mask": batch["mask"],
            }
        )

    configs = build_configs(args)
    print(
        f"[info] device={device} cases={len(case_cache)} num_samples={args.num_samples} "
        f"configs={len(configs)} weights={weights_name} protocol={args.metric_protocol}"
    )

    baseline_case_rows: list[dict[str, Any]] = []
    baseline_volume_rows: list[dict[str, Any]] = []

    zf_current_psnrs: list[float] = []
    zf_volume_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for c in case_cache:
        gt2 = c["target"][0]
        zf2 = c["cond"][0, :2]

        gt_mag = to_mag(gt2)
        zf_mag = to_mag(zf2)
        current_psnr_v = psnr_mag(zf_mag, gt_mag)
        zf_current_psnrs.append(current_psnr_v)

        if args.author_norm:
            gt_author_mag = author_prepare_mag_from_complex2(gt2, eps=float(args.author_eps))
            zf_author_mag = author_prepare_mag_from_complex2(zf2, eps=float(args.author_eps))
        else:
            gt_author_mag = gt_mag.detach().cpu().numpy().astype(np.float32)
            zf_author_mag = zf_mag.detach().cpu().numpy().astype(np.float32)

        zf_volume_buckets[c["volume_id"]].append(
            {
                "slice_idx": int(c["slice_idx"]),
                "case_idx": int(c["case_idx"]),
                "path": c["path"],
                "pred_author_mag": zf_author_mag,
                "gt_author_mag": gt_author_mag,
                "current_psnr": current_psnr_v,
            }
        )

        baseline_case_rows.append(
            {
                "case_idx": c["case_idx"],
                "volume_id": c["volume_id"],
                "slice_idx": c["slice_idx"],
                "path": c["path"],
                "current_psnr": current_psnr_v,
            }
        )

    zf_author_psnrs: list[float] = []
    for volume_id in sorted(zf_volume_buckets.keys()):
        items = sorted(zf_volume_buckets[volume_id], key=lambda x: x["slice_idx"])
        pred_stack = np.stack([it["pred_author_mag"] for it in items], axis=0)
        gt_stack = np.stack([it["gt_author_mag"] for it in items], axis=0)
        author_psnr_v, used_slices = author_volume_psnr(
            pred_stack_shw=pred_stack,
            gt_stack_shw=gt_stack,
            crop_head=int(args.author_crop_head),
            crop_tail=int(args.author_crop_tail),
            range_mode=str(args.author_range),
            eps=float(args.author_eps),
        )
        zf_author_psnrs.append(author_psnr_v)
        baseline_volume_rows.append(
            {
                "volume_id": volume_id,
                "num_slices_total": len(items),
                "num_slices_used": used_slices,
                "author_psnr": author_psnr_v,
            }
        )

    zf_current_psnr_mean, zf_current_psnr_std = summarize(zf_current_psnrs)
    zf_author_psnr_mean, zf_author_psnr_std = summarize(zf_author_psnrs)

    zf_baseline = {
        "name": "zf",
        "num_cases": len(case_cache),
        "num_volumes": len(baseline_volume_rows),
        "current_psnr_mean": zf_current_psnr_mean,
        "current_psnr_std": zf_current_psnr_std,
        "author_psnr_mean": zf_author_psnr_mean,
        "author_psnr_std": zf_author_psnr_std,
    }
    print(
        "[baseline zf] "
        f"current_psnr={zf_current_psnr_mean:.4f} "
        f"author_psnr={zf_author_psnr_mean:.4f} "
        f"volumes={len(baseline_volume_rows)}"
    )

    summary_rows: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    per_volume_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for cfg_id, cfg in enumerate(configs):
        current_psnrs: list[float] = []
        volume_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

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
                pred_author_mag = (
                    author_prepare_mag_from_complex2(pred2, eps=float(args.author_eps))
                    if args.author_norm
                    else pred_mag.detach().cpu().numpy().astype(np.float32)
                )
            else:
                pred_mag = torch.sqrt(stack[:, 0] ** 2 + stack[:, 1] ** 2 + 1e-12).mean(dim=0)
                pred_author_mag = (
                    author_prepare_mag_from_mag(pred_mag, eps=float(args.author_eps))
                    if args.author_norm
                    else pred_mag.detach().cpu().numpy().astype(np.float32)
                )

            gt_mag = to_mag(gt2)
            current_psnr_v = psnr_mag(pred_mag, gt_mag)
            current_psnrs.append(current_psnr_v)

            gt_author_mag = (
                author_prepare_mag_from_complex2(gt2, eps=float(args.author_eps))
                if args.author_norm
                else gt_mag.detach().cpu().numpy().astype(np.float32)
            )

            volume_buckets[c["volume_id"]].append(
                {
                    "slice_idx": int(c["slice_idx"]),
                    "case_idx": int(c["case_idx"]),
                    "path": c["path"],
                    "pred_author_mag": pred_author_mag,
                    "gt_author_mag": gt_author_mag,
                    "current_psnr": current_psnr_v,
                }
            )

            per_case_rows.append(
                {
                    "config_id": cfg_id,
                    "init_mode": cfg["init_mode"],
                    "init_blend": cfg["init_blend"],
                    "steps": cfg["steps"],
                    "dc": cfg["dc"],
                    "dc_start": cfg["dc_start"],
                    "dc_every": cfg["dc_every"],
                    "dc_lam": cfg["dc_lam"],
                    "dc_ramp": cfg["dc_ramp"],
                    "case_idx": c["case_idx"],
                    "volume_id": c["volume_id"],
                    "slice_idx": c["slice_idx"],
                    "path": c["path"],
                    "current_psnr": current_psnr_v,
                }
            )

        current_psnr_mean, current_psnr_std = summarize(current_psnrs)

        author_psnrs: list[float] = []
        for volume_id in sorted(volume_buckets.keys()):
            items = sorted(volume_buckets[volume_id], key=lambda x: x["slice_idx"])
            pred_stack = np.stack([it["pred_author_mag"] for it in items], axis=0)
            gt_stack = np.stack([it["gt_author_mag"] for it in items], axis=0)
            author_psnr_v, used_slices = author_volume_psnr(
                pred_stack_shw=pred_stack,
                gt_stack_shw=gt_stack,
                crop_head=int(args.author_crop_head),
                crop_tail=int(args.author_crop_tail),
                range_mode=str(args.author_range),
                eps=float(args.author_eps),
            )
            author_psnrs.append(author_psnr_v)
            per_volume_rows.append(
                {
                    "config_id": cfg_id,
                    "init_mode": cfg["init_mode"],
                    "init_blend": cfg["init_blend"],
                    "steps": cfg["steps"],
                    "dc": cfg["dc"],
                    "dc_start": cfg["dc_start"],
                    "dc_every": cfg["dc_every"],
                    "dc_lam": cfg["dc_lam"],
                    "dc_ramp": cfg["dc_ramp"],
                    "volume_id": volume_id,
                    "num_slices_total": len(items),
                    "num_slices_used": used_slices,
                    "author_psnr": author_psnr_v,
                }
            )

        author_psnr_mean, author_psnr_std = summarize(author_psnrs)

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
            "num_volumes": len(author_psnrs),
            "num_samples": int(args.num_samples),
            "aggregate": args.aggregate,
            "current_psnr_mean": current_psnr_mean,
            "current_psnr_std": current_psnr_std,
            "author_psnr_mean": author_psnr_mean,
            "author_psnr_std": author_psnr_std,
        }
        summary_rows.append(row)
        print(
            f"[cfg {cfg_id:03d}] init={cfg['init_mode']}"
            + (f"(a={cfg['init_blend']:.2f})" if cfg["init_blend"] is not None else "")
            + f" steps={cfg['steps']} dc={cfg['dc']} "
            f"start={cfg['dc_start']} every={cfg['dc_every']} lam={cfg['dc_lam']} ramp={cfg['dc_ramp']}  "
            f"current_psnr={current_psnr_mean:.4f} author_psnr={author_psnr_mean:.4f}"
        )

    rank_key = "author_psnr_mean" if args.rank_metric == "author_psnr" else "current_psnr_mean"
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: (-math.inf if np.isnan(r[rank_key]) else r[rank_key]),
        reverse=True,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "summary.csv"
    per_case_path = outdir / "per_case.csv"
    per_volume_path = outdir / "per_volume.csv"
    baseline_case_path = outdir / "baseline_zf_per_case.csv"
    baseline_volume_path = outdir / "baseline_zf_per_volume.csv"
    baseline_path = outdir / "baseline_zf.json"
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
    if len(per_volume_rows) > 0:
        with per_volume_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_volume_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_volume_rows)
    if len(baseline_case_rows) > 0:
        with baseline_case_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(baseline_case_rows[0].keys()))
            writer.writeheader()
            writer.writerows(baseline_case_rows)
    if len(baseline_volume_rows) > 0:
        with baseline_volume_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(baseline_volume_rows[0].keys()))
            writer.writeheader()
            writer.writerows(baseline_volume_rows)
    with baseline_path.open("w") as f:
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
        "metric_protocol": args.metric_protocol,
        "author_protocol": {
            "author_norm": bool(args.author_norm),
            "author_crop_head": int(args.author_crop_head),
            "author_crop_tail": int(args.author_crop_tail),
            "author_range": str(args.author_range),
            "author_eps": float(args.author_eps),
        },
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
    print(f"[done] wrote: {per_volume_path}")
    print(f"[done] wrote: {baseline_case_path}")
    print(f"[done] wrote: {baseline_volume_path}")
    print(f"[done] wrote: {baseline_path}")
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
