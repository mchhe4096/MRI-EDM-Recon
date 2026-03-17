import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.datasets import PtKspaceReconDataset
from src.models.unet_v2 import UNetV2
from src.diffusion.edm import EDMTrainer


def psnr_mag(pred2, gt2, eps=1e-12):
    pred = torch.sqrt(pred2[0]**2 + pred2[1]**2 + eps)
    gt = torch.sqrt(gt2[0]**2 + gt2[1]**2 + eps)
    mse = torch.mean((pred - gt) ** 2)
    maxv = torch.max(gt)
    return float((20 * torch.log10(maxv / torch.sqrt(mse + eps))).item())


def to_mag(x2: torch.Tensor) -> torch.Tensor:
    # x2: [2,H,W] -> [H,W]
    return torch.sqrt(x2[0] ** 2 + x2[1] ** 2 + 1e-12)


def save_img(path: str, img_hw: torch.Tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = img_hw.detach().cpu().float().numpy()

    # normalize for visualization
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)
    img = np.clip((img - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)

    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="../outputs/ckpts/last.pt")
    p.add_argument("--val_root", type=str, default="../data/val")

    # dataset params (must match train)
    p.add_argument("--accel", type=int, default=4)
    p.add_argument("--center_frac", type=float, default=0.08)
    p.add_argument(
        "--include_mask_channel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override include_mask_channel from checkpoint args.",
    )
    p.add_argument("--base_ch", type=int, default=None, help="Override UNet base channels from checkpoint args.")

    # sampling params
    p.add_argument("--num_cases", type=int, default=8, help="Number of validation cases to sample. Use 0 for all.")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--dc", action="store_true")
    p.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EMA weights from checkpoint for sampling.",
    )
    p.add_argument("--save_pt", action=argparse.BooleanOptionalAction, default=True,
                   help="Save per-case result.pt for metric evaluation.")
    p.add_argument("--save_all_samples_pt", action="store_true",
                   help="Also save all sampled reconstructions in result.pt (larger disk usage).")

    # --- new: DC schedule (soft DC) ---
    p.add_argument("--dc_start", type=float, default=0.6)   # start ratio, steps=20 -> i>=12
    p.add_argument("--dc_every", type=int, default=2)       # apply every N steps after start
    p.add_argument("--dc_lam", type=float, default=0.15)    # base lambda for soft DC
    p.add_argument("--dc_ramp", action="store_true")        # ramp lam to 0.25 towards the end
    p.add_argument(
        "--init_mode",
        type=str,
        default="zf",
        choices=["noise", "zf", "blend"],
        help="Sampler initialization mode.",
    )
    p.add_argument(
        "--init_blend",
        type=float,
        default=0.5,
        help="Blend alpha used when --init_mode=blend (alpha*x_zf + (1-alpha)*noise).",
    )
    p.add_argument("--sigma_data", type=float, default=None, help="Override sigma_data from checkpoint args.")

    p.add_argument("--outdir", type=str, default="../outputs/samples")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {args.ckpt}\n"
            "This checkpoint may be corrupted or incomplete. "
            "Please use a valid step_*.pt/last.pt and retry."
        ) from e
    train_args = ckpt.get("args", {})

    sigma_min = float(train_args.get("sigma_min", 0.002))
    sigma_max = float(train_args.get("sigma_max", 80.0))
    if "sigma_data" not in train_args and args.sigma_data is None:
        raise ValueError(
            "Checkpoint does not contain sigma_data. Old checkpoint format is not supported anymore. "
            "Please retrain with the new EDM preconditioning objective."
        )
    sigma_data = float(train_args.get("sigma_data", 0.5)) if args.sigma_data is None else float(args.sigma_data)
    if train_args.get("precondition", True) is False:
        raise ValueError(
            "Checkpoint was trained with legacy non-precondition objective, which is not supported anymore."
        )
    base_ch = int(train_args.get("base_ch", 128)) if args.base_ch is None else int(args.base_ch)
    include_mask_channel = bool(
        train_args.get("include_mask_channel", True)
        if args.include_mask_channel is None
        else args.include_mask_channel
    )

    cond_ch = 3 if include_mask_channel else 2
    model = UNetV2(x_ch=2, cond_ch=cond_ch, out_ch=2, base_ch=base_ch).to(device)
    if args.use_ema:
        if "ema_model" not in ckpt:
            raise ValueError(
                "Checkpoint does not contain ema_model. Use a new checkpoint with EMA or run with --no-use_ema."
            )
        weights = ckpt["ema_model"]
        loaded_weights = "ema_model"
    else:
        weights = ckpt["model"]
        loaded_weights = "model"
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "Checkpoint is incompatible with current UNetV2 architecture. "
            "Please retrain to get a new checkpoint after the sigma/time-embedding upgrade."
        ) from e
    model.eval()

    trainer = EDMTrainer(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
    )

    ds = PtKspaceReconDataset(
        root=args.val_root,
        accel=args.accel,
        center_frac=args.center_frac,
        include_mask_channel=include_mask_channel,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    total_cases = len(ds) if args.num_cases == 0 else min(args.num_cases, len(ds))
    print(
        f"[info] sampling {total_cases}/{len(ds)} cases from {args.val_root}  "
        f"(sigma_data={sigma_data:.4g}, base_ch={base_ch}, weights={loaded_weights}, "
        f"include_mask_channel={include_mask_channel}, init={args.init_mode}"
        + (f"(a={args.init_blend:.2f})" if args.init_mode == "blend" else "")
        + ")"
    )

    for case_idx, batch in enumerate(loader):
        if case_idx >= total_cases:
            break

        # NOTE: 你们已经把 dataset 改成返回 k_us 了（按你说的）
        target = batch["target"].to(device)        # [1,2,H,W]
        cond = batch["cond"].to(device)            # [1,C,H,W]
        mask = batch["mask"].to(device)            # [1,1,H,W]
        k_us = batch.get("k_us", None)
        if k_us is not None:
            k_us = k_us.to(device)                # [1,2,H,W]

        # cond 里前两通道是 x_zf（你们现在就是这么拼的）
        x_zf = cond[:, :2]

        case_dir = os.path.join(args.outdir, f"case_{case_idx:03d}")
        os.makedirs(case_dir, exist_ok=True)

        save_img(os.path.join(case_dir, "zf.png"), to_mag(x_zf[0]))
        save_img(os.path.join(case_dir, "gt.png"), to_mag(target[0]))

        samples = []
        for n in range(args.num_samples):
            x = trainer.sample(
                model=model,
                shape=target.shape,       # [1,2,H,W]
                cond=cond,
                steps=args.steps,
                dc=args.dc,
                k_us=k_us,
                mask=mask,

                # --- new: DC schedule (must be supported by trainer.sample -> sampler) ---
                dc_start=args.dc_start,
                dc_every=args.dc_every,
                dc_lam=args.dc_lam,
                dc_ramp=args.dc_ramp,
                init_mode=args.init_mode,
                init_blend=float(args.init_blend),
            )
            samples.append(x[0].detach().cpu())
            save_img(os.path.join(case_dir, f"sample_{n:02d}.png"), to_mag(x[0]))

        stack = torch.stack(samples, dim=0)               # [N,2,H,W]
        mean2 = stack.mean(dim=0)  # [2,H,W] 复数均值
        mags = torch.sqrt(stack[:, 0] ** 2 + stack[:, 1] ** 2 + 1e-12)
        std = mags.std(dim=0, unbiased=False)
        mean = torch.sqrt(mean2[0]**2 + mean2[1]**2 + 1e-12)  # [H,W]

        save_img(os.path.join(case_dir, "mean.png"), mean)
        save_img(os.path.join(case_dir, "std.png"), std)
        case_psnr = psnr_mag(mean2, target[0].detach().cpu())

        if args.save_pt:
            path_item = batch.get("path", "")
            if isinstance(path_item, (list, tuple)):
                path_item = path_item[0] if len(path_item) > 0 else ""
            result = {
                "case_idx": case_idx,
                "path": str(path_item),
                "psnr_mean_db": case_psnr,
                "mean2": mean2,                        # [2,H,W]
                "std": std.detach().cpu(),             # [H,W]
                "target": target[0].detach().cpu(),    # [2,H,W]
                "x_zf": x_zf[0].detach().cpu(),        # [2,H,W]
                "mask": mask[0].detach().cpu(),        # [1,H,W]
                "init_mode": args.init_mode,
                "init_blend": float(args.init_blend),
                "include_mask_channel": include_mask_channel,
            }
            if args.save_all_samples_pt:
                result["samples"] = stack              # [N,2,H,W]
            torch.save(result, os.path.join(case_dir, "result.pt"))

        print(
            f"[case {case_idx:03d}] psnr(mean)={case_psnr:.2f} dB  saved to {case_dir}  "
            f"dc={args.dc} init={args.init_mode}"
            + (f"(a={args.init_blend:.2f})" if args.init_mode == "blend" else "")
        )


if __name__ == "__main__":
    main()
