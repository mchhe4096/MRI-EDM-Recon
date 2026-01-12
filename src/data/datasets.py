from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import complex_to_2ch
from src.mri.mask import cartesian_mask
from src.mri.operators import ifft2c

class PtComplexImageDataset(Dataset):
    """
    Each .pt is actually a pickled dict with key 'img' -> np.complex64 [H,W].
    Returns:
      x_gt: [2,H,W] float32
    """
    def __init__(self, root: str):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.pt"))
        assert len(self.files) > 0, f"No .pt files found under {root}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with open(path, "rb") as f:
            obj = pickle.load(f)
        img = obj["img"]
        assert isinstance(img, np.ndarray) and np.iscomplexobj(img)
        x_gt = complex_to_2ch(img)  # [2,H,W]
        return {"x_gt": x_gt, "path": str(path)}


class PtKspaceReconDataset(Dataset):
    """
    Expect torch-saved dict per slice:
      - kspace_full: [H,W,2] float or [2,H,W] float
      - img_gt:      [H,W,2] float or [2,H,W] float  (optional; can be ifft2c(kspace_full))
      - (optional) fname, slice_id, scale_k
    Returns:
      target: [2,H,W]
      cond:   [C,H,W]  where C=2 (x_zf) or C=3 (x_zf + mask)
      mask:   [1,H,W]
    """
    def __init__(self, root: str, accel=4, center_frac=0.08, include_mask_channel=True):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.pt"))
        assert len(self.files) > 0, f"No .pt files found under {root}"
        self.accel = accel
        self.center_frac = center_frac
        self.include_mask_channel = include_mask_channel

    def __len__(self):
        return len(self.files)

    def _ensure_2chw(self, t: torch.Tensor) -> torch.Tensor:
        # Accept [H,W,2] or [2,H,W]
        if t.ndim == 3 and t.shape[-1] == 2:
            t = t.permute(2, 0, 1).contiguous()
        assert t.ndim == 3 and t.shape[0] == 2
        return t.float()

    def __getitem__(self, idx: int):
        path = self.files[idx]
        d = torch.load(path, map_location="cpu", weights_only=True)

        k_full = self._ensure_2chw(d["kspace_full"])  # [2,H,W]
        H, W = k_full.shape[1], k_full.shape[2]

        mask = cartesian_mask((H, W), accel=self.accel, center_frac=self.center_frac)  # [1,1,H,W]
        k_us = k_full.unsqueeze(0) * mask  # [1,2,H,W] broadcast on channel ok
        x_zf = ifft2c(k_us).squeeze(0)     # [2,H,W]

        if "img_gt" in d:
            target = self._ensure_2chw(d["img_gt"])
        else:
            target = ifft2c(k_full.unsqueeze(0)).squeeze(0)

        # conditioning A
        if self.include_mask_channel:
            cond = torch.cat([x_zf, mask.squeeze(0)], dim=0)  # [3,H,W]
        else:
            cond = x_zf  # [2,H,W]

        return {
            "target": target,
            "cond": cond,
            "mask": mask.squeeze(0),  # [1,H,W]
            "k_us": k_us.squeeze(0),      # [2,H,W]  观测（关键！）
            "k_full": k_full,             # [2,H,W]  可选：用于 debug/对比
            "path": str(path),
        }