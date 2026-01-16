import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.data.datasets import PtKspaceReconDataset
from src.models.unet import SimpleUNetStub
from src.diffusion.edm import EDMTrainer


def train_one_epoch(model, trainer, loader, optim, device):
    model.train()
    total = 0.0
    for batch in loader:
        x0 = batch["target"].to(device)      # [B,2,H,W]
        cond = batch["cond"].to(device)      # [B,C,H,W]

        loss = trainer.loss(model, x0, cond=cond)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def eval_loss(model, trainer, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        x0 = batch["target"].to(device)
        cond = batch["cond"].to(device)
        loss = trainer.loss(model, x0, cond=cond)
        total += float(loss.item())
    return total / max(1, len(loader))


def save_ckpt(path, model, optim, epoch, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", type=str, default="../debug_data/train")
    p.add_argument("--val_root", type=str, default="../debug_data/val")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)

    # task / conditioning
    p.add_argument("--accel", type=int, default=4)
    p.add_argument("--center_frac", type=float, default=0.08)
    p.add_argument("--include_mask_channel", action="store_true")

    # diffusion-ish
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)

    # ckpt
    p.add_argument("--outdir", type=str, default="../outputs/ckpts")

    args = p.parse_args()
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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    cond_ch = 3 if args.include_mask_channel else 2
    model = SimpleUNetStub(x_ch=2, cond_ch=cond_ch, out_ch=2).to(device)
    trainer = EDMTrainer(sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, trainer, train_loader, optim, device)
        va = eval_loss(model, trainer, val_loader, device)
        print(f"[epoch {epoch:03d}] train_loss={tr:.6f}  val_loss={va:.6f}")

        ckpt_path = os.path.join(args.outdir, f"epoch_{epoch:03d}.pt")
        save_ckpt(ckpt_path, model, optim, epoch, args)
        save_ckpt(os.path.join(args.outdir, "last.pt"), model, optim, epoch, args)


if __name__ == "__main__":
    main()
