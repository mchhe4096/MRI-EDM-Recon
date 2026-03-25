from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.diffusion.edm import EDMTrainer
from src.models.unet_v2 import UNetV2
from src.mri.mask import cartesian_mask
from src.mri.operators import ifft2c


def _ensure_2chw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3 and t.shape[-1] == 2:
        t = t.permute(2, 0, 1).contiguous()
    if t.ndim != 3 or t.shape[0] != 2:
        raise ValueError(f"Expected tensor shape [2,H,W] or [H,W,2], got {tuple(t.shape)}")
    return t.float()


def _to_mag(x2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(x2[0] * x2[0] + x2[1] * x2[1] + eps)


def _to_uint8(img_hw: torch.Tensor) -> np.ndarray:
    arr = img_hw.detach().cpu().float().numpy()
    p1 = np.percentile(arr, 1.0)
    p99 = np.percentile(arr, 99.0)
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-12), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


@lru_cache(maxsize=4)
def _load_bundle(
    ckpt_path: str,
    device_str: str,
    use_ema: bool,
    include_mask_channel_mode: str,
):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})

    sigma_min = float(train_args.get("sigma_min", 0.002))
    sigma_max = float(train_args.get("sigma_max", 80.0))
    sigma_data = float(train_args.get("sigma_data", 0.5))
    base_ch = int(train_args.get("base_ch", 128))
    accel = int(train_args.get("accel", 4))
    center_frac = float(train_args.get("center_frac", 0.08))

    if include_mask_channel_mode == "auto":
        include_mask_channel = bool(train_args.get("include_mask_channel", True))
    elif include_mask_channel_mode == "true":
        include_mask_channel = True
    elif include_mask_channel_mode == "false":
        include_mask_channel = False
    else:
        raise ValueError(f"Invalid include_mask_channel_mode: {include_mask_channel_mode}")

    device = torch.device(device_str)
    cond_ch = 3 if include_mask_channel else 2
    model = UNetV2(x_ch=2, cond_ch=cond_ch, out_ch=2, base_ch=base_ch).to(device)

    if use_ema:
        if "ema_model" not in ckpt:
            raise ValueError("Checkpoint does not contain ema_model. Disable use_ema or use a newer checkpoint.")
        weights = ckpt["ema_model"]
        weights_name = "ema_model"
    else:
        weights = ckpt["model"]
        weights_name = "model"

    model.load_state_dict(weights, strict=True)
    model.eval()

    trainer = EDMTrainer(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
    )
    return model, trainer, include_mask_channel, accel, center_frac, weights_name, device


@torch.no_grad()
def reconstruct_from_pt(
    pt_path: str,
    ckpt_path: str,
    steps: int,
    dc: bool,
    init_mode: str,
    init_blend: float,
    dc_start: float,
    dc_every: int,
    dc_lam: float,
    dc_ramp: bool,
    use_ema: bool,
    include_mask_channel_mode: str = "auto",
):
    pt_path = str(Path(pt_path).expanduser().resolve())
    ckpt_path = str(Path(ckpt_path).expanduser().resolve())

    if not Path(pt_path).exists():
        raise FileNotFoundError(f"Input file not found: {pt_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model, trainer, include_mask_channel, accel, center_frac, weights_name, device = _load_bundle(
        ckpt_path=ckpt_path,
        device_str=device_str,
        use_ema=bool(use_ema),
        include_mask_channel_mode=include_mask_channel_mode,
    )

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "kspace_full" not in data:
        raise KeyError(f"'kspace_full' is missing in {pt_path}")
    k_full = _ensure_2chw(data["kspace_full"])
    h, w = int(k_full.shape[1]), int(k_full.shape[2])

    mask = cartesian_mask((h, w), accel=accel, center_frac=center_frac)  # [1,1,H,W]
    k_us = k_full.unsqueeze(0) * mask
    x_zf = ifft2c(k_us).squeeze(0)

    if include_mask_channel:
        cond = torch.cat([x_zf, mask.squeeze(0)], dim=0).unsqueeze(0)
    else:
        cond = x_zf.unsqueeze(0)
    cond = cond.to(device)

    target = None
    if "img_gt" in data:
        target = _ensure_2chw(data["img_gt"])

    recon = trainer.sample(
        model=model,
        shape=(1, 2, h, w),
        cond=cond,
        steps=int(steps),
        dc=bool(dc),
        k_us=k_us.to(device),
        mask=mask.to(device),
        dc_start=float(dc_start),
        dc_every=int(dc_every),
        dc_lam=float(dc_lam),
        dc_ramp=bool(dc_ramp),
        init_mode=str(init_mode),
        init_blend=float(init_blend),
    )[0].detach().cpu()

    recon_mag = _to_mag(recon)
    zf_mag = _to_mag(x_zf)
    gt_mag = _to_mag(target) if target is not None else None

    info_lines = [
        f"device={device}",
        f"weights={weights_name}",
        f"include_mask_channel={include_mask_channel}",
        f"shape={tuple(recon.shape)}",
        f"accel={accel}, center_frac={center_frac}",
        f"steps={int(steps)}, dc={bool(dc)}, init={init_mode}, init_blend={float(init_blend):.2f}",
    ]
    if target is not None:
        mse = torch.mean((recon_mag - gt_mag) ** 2)
        maxv = torch.max(gt_mag).clamp_min(1e-12)
        psnr = float((20.0 * torch.log10(maxv / torch.sqrt(mse + 1e-12))).item())
        info_lines.append(f"psnr(recon, gt)={psnr:.2f} dB")

    return _to_uint8(recon_mag), _to_uint8(zf_mag), (_to_uint8(gt_mag) if gt_mag is not None else None), "\n".join(
        info_lines
    )

