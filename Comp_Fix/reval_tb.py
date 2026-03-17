import argparse
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data.datasets import PtKspaceReconDataset
from src.diffusion.edm import EDMTrainer
from src.diffusion.objective import edm_denoise
from src.models.unet_v2 import UNetV2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-evaluate checkpoints and write corrected TensorBoard curves.")
    p.add_argument("--ckpt_dir", type=str, default="../outputs/ckpts")
    p.add_argument("--glob", type=str, default="step_*.pt")
    p.add_argument("--include_last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--val_root", type=str, default="/root/autodl-tmp/val_v2")
    p.add_argument("--val_subset", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=20)
    p.add_argument("--sigma_proxy", type=float, default=1.0)

    p.add_argument("--accel", type=int, default=None)
    p.add_argument("--center_frac", type=float, default=None)
    p.add_argument("--include_mask_channel", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--base_ch", type=int, default=None)
    p.add_argument("--sigma_min", type=float, default=None)
    p.add_argument("--sigma_max", type=float, default=None)
    p.add_argument("--sigma_data", type=float, default=None)

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tb_root", type=str, default="/root/tf-logs")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def _parse_step_from_name(path: Path) -> int | None:
    m = re.match(r"step_(\d+)\.pt$", path.name)
    if m is None:
        return None
    return int(m.group(1))


def _list_checkpoints(ckpt_dir: Path, glob_pat: str, include_last: bool) -> list[Path]:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

    step_map: dict[int, Path] = {}
    for p in sorted(ckpt_dir.glob(glob_pat)):
        step = _parse_step_from_name(p)
        if step is not None:
            step_map[step] = p

    if include_last:
        last_path = ckpt_dir / "last.pt"
        if last_path.exists():
            ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
            step = int(ckpt.get("step", -1))
            if step >= 0 and step not in step_map:
                step_map[step] = last_path

    if len(step_map) == 0:
        raise FileNotFoundError(f"no checkpoints found under {ckpt_dir} with glob={glob_pat}")

    return [step_map[s] for s in sorted(step_map.keys())]


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred_mag = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + eps)
    tgt_mag = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2 + eps)
    mse = torch.mean((pred_mag - tgt_mag) ** 2)
    maxv = torch.max(tgt_mag)
    return 20 * torch.log10(maxv / torch.sqrt(mse + eps))


@torch.no_grad()
def val_metrics(
    model: torch.nn.Module,
    trainer: EDMTrainer,
    loader: DataLoader,
    device: torch.device,
    sigma_proxy: float,
    max_batches: int,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    psnrs: list[float] = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x0 = batch["target"].to(device)
        cond = batch["cond"].to(device)

        # Keep the same loss definition as training.
        loss = trainer.loss(model, x0, cond)

        # Correct proxy: convert model output to x0 via EDM preconditioning.
        bsz = x0.shape[0]
        sigma = torch.full((bsz, 1, 1, 1), float(sigma_proxy), device=device)
        x_noisy = x0 + sigma * torch.randn_like(x0)
        x0_pred, _ = edm_denoise(model, x_noisy, sigma, cond, sigma_data=trainer.sigma_data)

        losses.append(float(loss.item()))
        psnrs.append(float(psnr(x0_pred, x0).item()))

    return float(np.mean(losses)), float(np.mean(psnrs))


def _resolve(v_cli, args_ckpt: dict, key: str, default):
    if v_cli is not None:
        return v_cli
    return args_ckpt.get(key, default)


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_paths = _list_checkpoints(ckpt_dir, args.glob, include_last=bool(args.include_last))

    first_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    first_args = first_ckpt.get("args", {})

    accel = int(_resolve(args.accel, first_args, "accel", 4))
    center_frac = float(_resolve(args.center_frac, first_args, "center_frac", 0.08))
    include_mask_channel = bool(_resolve(args.include_mask_channel, first_args, "include_mask_channel", True))
    base_ch = int(_resolve(args.base_ch, first_args, "base_ch", 128))
    sigma_min = float(_resolve(args.sigma_min, first_args, "sigma_min", 0.002))
    sigma_max = float(_resolve(args.sigma_max, first_args, "sigma_max", 80.0))
    sigma_data = float(_resolve(args.sigma_data, first_args, "sigma_data", 0.5))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("[warn] running reval on CPU; this can be slow.")

    val_ds = PtKspaceReconDataset(
        root=args.val_root,
        accel=accel,
        center_frac=center_frac,
        include_mask_channel=include_mask_channel,
    )
    if args.val_subset and args.val_subset > 0:
        val_ds = Subset(val_ds, list(range(min(args.val_subset, len(val_ds)))))

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    cond_ch = 3 if include_mask_channel else 2
    model = UNetV2(x_ch=2, cond_ch=cond_ch, out_ch=2, base_ch=base_ch).to(device)
    trainer = EDMTrainer(sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data)

    tb_root = Path(args.tb_root)
    tb_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name if args.run_name else f"reval_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(tb_root / run_name))

    print(f"[info] checkpoints={len(ckpt_paths)} val_size={len(val_ds)} device={device}")
    print(f"[info] writing TensorBoard run: {tb_root / run_name}")
    print(
        "[info] config "
        f"accel={accel} center_frac={center_frac} include_mask_channel={include_mask_channel} "
        f"base_ch={base_ch} sigma_min={sigma_min} sigma_max={sigma_max} sigma_data={sigma_data}"
    )

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        step = int(ckpt.get("step", _parse_step_from_name(ckpt_path) or 0))

        weights_key = "ema_model" if (args.use_ema and "ema_model" in ckpt) else "model"
        model.load_state_dict(ckpt[weights_key], strict=True)

        # Make random components deterministic for each checkpoint.
        seed = int(args.seed + step)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        vloss, vpsnr = val_metrics(
            model=model,
            trainer=trainer,
            loader=val_loader,
            device=device,
            sigma_proxy=float(args.sigma_proxy),
            max_batches=int(args.max_batches),
        )

        writer.add_scalar("val/loss", vloss, step)
        writer.add_scalar("val/psnr_proxy", vpsnr, step)
        writer.flush()
        print(
            f"[reval] step={step:07d} "
            f"val/loss={vloss:.6f} val/psnr_proxy={vpsnr:.4f} "
            f"weights={weights_key} ckpt={ckpt_path.name}"
        )

    writer.close()
    print("[done] re-evaluation finished.")


if __name__ == "__main__":
    main()
