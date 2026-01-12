import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(model, trainer, loader, optim, device):
    model.train()
    total = 0.0
    for batch in loader:   # ✅ loader
        x0 = batch["target"].to(device)
        cond = batch["cond"].to(device)

        loss = trainer.loss(model, x0, cond=cond)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total += float(loss.item())

    return total / max(1, len(loader))

