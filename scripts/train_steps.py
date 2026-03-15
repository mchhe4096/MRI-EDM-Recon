import os
import time
import argparse
import random
import math
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data.datasets import PtKspaceReconDataset
from src.models.unet_v2 import UNetV2
from src.diffusion.edm import EDMTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def create_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    # Update parameters with EMA; copy buffers directly.
    ema_params = dict(ema_model.named_parameters())
    for name, p in model.named_parameters():
        ema_params[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    for name, b in model.named_buffers():
        ema_buffers[name].copy_(b.detach())


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if total_steps <= 1:
        return 1.0

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))

    if total_steps <= warmup_steps:
        return 1.0

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def set_optimizer_lr(optim: torch.optim.Optimizer, lr: float):
    for pg in optim.param_groups:
        pg["lr"] = lr


def save_ckpt(path, model, ema_model, optim, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optim": optim.state_dict(),
        "args": vars(args),
    }
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


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
    p.add_argument("--train_root", type=str, default="../data/train")
    p.add_argument("--val_root", type=str, default="../data/val")
    p.add_argument("--train_subset", type=int, default=0, help="0 means full dataset; otherwise use first N samples")
    p.add_argument("--val_subset", type=int, default=0, help="0 means full; otherwise use first N for quick val")

    # task
    p.add_argument("--accel", type=int, default=4)
    p.add_argument("--center_frac", type=float, default=0.08)
    p.add_argument("--include_mask_channel", action="store_true")

    # training
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.05, help="Final LR = lr * min_lr_ratio.")
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--num_workers", type=int, default=0)  # Windows 建议 0~2
    p.add_argument("--seed", type=int, default=0)

    # diffusion params
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--sigma_data", type=float, default=0.5)

    # logging/ckpt
    p.add_argument("--outdir", type=str, default="../outputs/ckpts")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="", help="path to ckpt to resume, e.g., ../outputs/ckpts/last.pt")

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
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True,persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    cond_ch = 3 if args.include_mask_channel else 2
    model = UNetV2(x_ch=2, cond_ch=cond_ch, out_ch=2, base_ch=128).to(device)
    ema_model = create_ema_model(model)
    trainer = EDMTrainer(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
    )

    logdir = "/root/tf-logs"
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(logdir, run_name))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        try:
            model.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                "Resume checkpoint is incompatible with current model architecture. "
                "Please start a fresh training run after the UNetV2 sigma/time-embedding upgrade."
            ) from e
        if "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"], strict=True)
        else:
            ema_model.load_state_dict(ckpt["model"], strict=True)
            print("[warn] resume checkpoint has no ema_model; initialized EMA from model weights")
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

        cur_lr = args.lr * lr_multiplier(
            step=step,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        set_optimizer_lr(optim, cur_lr)

        model.train()
        loss = trainer.loss(model, x0, cond)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr", cur_lr, step)

        # Non-finite guard before backward.
        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at step {step+1}, skip")
            continue

        optim.zero_grad(set_to_none=True)
        loss.backward()

        # 1) 梯度裁剪，防止偶发爆炸把权重推崩
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()
        decay = min(float(args.ema_decay), (1.0 + step) / (10.0 + step))
        ema_update(ema_model, model, decay=decay)
        writer.add_scalar("train/ema_decay", decay, step)

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"[step {step+1:07d}] loss={loss.item():.6f} lr={cur_lr:.3e} "
                f"ema_decay={decay:.6f} ({dt/args.log_every:.3f}s/iter)"
            )

        if (step + 1) % args.val_every == 0:
            vloss, vpsnr = val_metrics(ema_model, trainer, val_loader, device, max_batches=20)
            writer.add_scalar("val/loss", vloss, step + 1)
            writer.add_scalar("val/psnr_proxy", vpsnr, step + 1)
            print(f"[val(ema) @ {step+1:07d}] loss={vloss:.6f}  psnr(proxy)={vpsnr:.2f}dB")

        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(os.path.join(args.outdir, f"step_{step+1:07d}.pt"), model, ema_model, optim, step + 1, args)
            save_ckpt(os.path.join(args.outdir, "last.pt"), model, ema_model, optim, step + 1, args)
            print(f"[ckpt] saved at step {step+1:07d}")

    save_ckpt(os.path.join(args.outdir, "last.pt"), model, ema_model, optim, args.steps, args)
    print("[done] training finished")


if __name__ == "__main__":
    main()
