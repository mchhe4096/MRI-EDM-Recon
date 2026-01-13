import os
import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.datasets import PtKspaceReconDataset
from src.models.unet import SimpleUNetStub
from src.diffusion.edm import EDMTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path, model, optim, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"step": step, "model": model.state_dict(), "optim": optim.state_dict(), "args": vars(args)},
        path
    )


@torch.no_grad()
def psnr(pred, target, eps=1e-12):
    # pred/target: [B,2,H,W] (complex as 2ch)
    pred_mag = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + eps)
    tgt_mag = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2 + eps)
    mse = torch.mean((pred_mag - tgt_mag) ** 2)
    maxv = torch.max(tgt_mag)
    return 20 * torch.log10(maxv / torch.sqrt(mse + eps))


@torch.no_grad()
def val_metrics(model, trainer, loader, device, max_batches=20):
    model.eval()
    losses = []
    psnrs = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x0 = batch["target"].to(device)
        cond = batch["cond"].to(device)
        loss = trainer.loss(model, x0, cond)
        # quick single-step "denoise" proxy for sanity: predict x0 from a noisy x
        # (not a full sampling; cheap)
        B = x0.shape[0]
        sigma = torch.full((B, 1, 1, 1), 1.0, device=device)
        x_noisy = x0 + sigma * torch.randn_like(x0)
        x0_pred = model(x_noisy, sigma, cond)
        losses.append(float(loss.item()))
        psnrs.append(float(psnr(x0_pred, x0).item()))
    return float(np.mean(losses)), float(np.mean(psnrs))


def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train_root", type=str, default="../debug_data/train")
    p.add_argument("--val_root", type=str, default="../debug_data/val")
    p.add_argument("--train_subset", type=int, default=0, help="0 means full dataset; otherwise use first N samples")
    p.add_argument("--val_subset", type=int, default=256, help="0 means full; otherwise use first N for quick val")

    # task
    p.add_argument("--accel", type=int, default=4)
    p.add_argument("--center_frac", type=float, default=0.08)
    p.add_argument("--include_mask_channel", action="store_true")

    # training
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)  # Windows 建议 0~2
    p.add_argument("--seed", type=int, default=0)

    # diffusion params
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)

    # logging/ckpt
    p.add_argument("--outdir", type=str, default="outputs/ckpts")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="", help="path to ckpt to resume, e.g., outputs/ckpts/last.pt")

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PtKspaceReconDataset(
        root=args.train_root,
        accel=args.accel,
        center_frac=args.center_frac,
        include_mask_channel=args.include_mask_channel,
    )
    val_ds = PtKspaceReconDataset(
        root=args.val_root,
        accel=args.accel,
        center_frac=args.center_frac,
        include_mask_channel=args.include_mask_channel,
    )

    if args.train_subset and args.train_subset > 0:
        train_ds = Subset(train_ds, list(range(min(args.train_subset, len(train_ds)))))
    if args.val_subset and args.val_subset > 0:
        val_ds = Subset(val_ds, list(range(min(args.val_subset, len(val_ds)))))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    cond_ch = 3 if args.include_mask_channel else 2
    model = SimpleUNetStub(x_ch=2, cond_ch=cond_ch, out_ch=2).to(device)
    trainer = EDMTrainer(sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optim.load_state_dict(ckpt["optim"])
        start_step = int(ckpt.get("step", 0))
        print(f"[resume] step={start_step} from {args.resume}")

    os.makedirs(args.outdir, exist_ok=True)

    it = iter(train_loader)
    t0 = time.time()
    for step in range(start_step, args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        x0 = batch["target"].to(device)
        cond = batch["cond"].to(device)

        model.train()
        loss = trainer.loss(model, x0, cond)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"[step {step+1:07d}] loss={loss.item():.6f}  ({dt/args.log_every:.3f}s/iter)")

        if (step + 1) % args.val_every == 0:
            vloss, vpsnr = val_metrics(model, trainer, val_loader, device, max_batches=20)
            print(f"[val @ {step+1:07d}] loss={vloss:.6f}  psnr(proxy)={vpsnr:.2f}dB")

        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(os.path.join(args.outdir, f"step_{step+1:07d}.pt"), model, optim, step + 1, args)
            save_ckpt(os.path.join(args.outdir, "last.pt"), model, optim, step + 1, args)
            print(f"[ckpt] saved at step {step+1:07d}")

    save_ckpt(os.path.join(args.outdir, "last.pt"), model, optim, args.steps, args)
    print("[done] training finished")


if __name__ == "__main__":
    main()
