import numpy as np
import torch

def complex_to_2ch(x: np.ndarray) -> torch.Tensor:
    """x: np.complex64 [H,W] -> torch.float32 [2,H,W]"""
    assert np.iscomplexobj(x), "expected complex ndarray"
    xr = np.stack([x.real, x.imag], axis=0)  # [2,H,W]
    return torch.from_numpy(xr).float()

def ch2_to_complex(x: torch.Tensor) -> torch.Tensor:
    """x: torch [2,H,W] -> torch.complex64 [H,W]"""
    assert x.ndim == 3 and x.shape[0] == 2
    return torch.complex(x[0], x[1])