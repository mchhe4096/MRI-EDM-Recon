import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.datasets import PtKspaceReconDataset
from src.models.unet import SimpleUNetStub
from src.diffusion.edm import EDMTrainer


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
    p.add_argument("--val_root", type=str, default="../debug_data/val")

    # dataset params (must match train)
    p.add_argument("--accel", type=int, default=4)
    p.add_argument("--center_frac", type=float, default=0.08)
    p.add_argument("--include_mask_channel", action="store_true")

    # sampling params
    p.add_argument("--num_cases", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--step_size", type=float, default=0.1)
    p.add_argument("--dc", action="store_true")

    p.add_argument("--outdir", type=str, default="../outputs/samples")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})

    sigma_min = float(train_args.get("sigma_min", 0.002))
    sigma_max = float(train_args.get("sigma_max", 80.0))

    cond_ch = 3 if args.include_mask_channel else 2
    model = SimpleUNetStub(x_ch=2, cond_ch=cond_ch, out_ch=2).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    trainer = EDMTrainer(sigma_min=sigma_min, sigma_max=sigma_max)

    ds = PtKspaceReconDataset(
        root=args.val_root,
        accel=args.accel,
        center_frac=args.center_frac,
        include_mask_channel=args.include_mask_channel,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for case_idx, batch in enumerate(loader):
        if case_idx >= args.num_cases:
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
                step_size=args.step_size,
            )
            samples.append(x[0].detach().cpu())
            save_img(os.path.join(case_dir, f"sample_{n:02d}.png"), to_mag(x[0]))

        stack = torch.stack(samples, dim=0)               # [N,2,H,W]
        mags = torch.sqrt(stack[:, 0] ** 2 + stack[:, 1] ** 2 + 1e-12)  # [N,H,W]
        mean = mags.mean(dim=0)
        std = mags.std(dim=0)

        save_img(os.path.join(case_dir, "mean.png"), mean)
        save_img(os.path.join(case_dir, "std.png"), std)

        print(f"[case {case_idx:03d}] saved to {case_dir}  dc={args.dc}")

if __name__ == "__main__":
    main()
