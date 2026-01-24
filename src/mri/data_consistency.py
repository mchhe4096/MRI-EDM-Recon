import torch
from .operators import fft2c, ifft2c

@torch.no_grad()
def dc_hard_kspace(x: torch.Tensor, k_us: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x:    [B,2,H,W]  当前图像域样本
    k_us: [B,2,H,W]  欠采样观测kspace（已乘mask）
    mask: [B,1,H,W]  {0,1}
    """
    k_pred = fft2c(x)                                # [B,2,H,W]
    mask2 = mask.expand(-1, 2, -1, -1)               # [B,2,H,W]
    k_new = k_pred * (1 - mask2) + k_us * mask2      # 已采样点替换成观测
    x_new = ifft2c(k_new)
    return x_new

@torch.no_grad()
def dc_soft_kspace(
    x: torch.Tensor,
    k_us: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 0.15,
) -> torch.Tensor:
    """
    Soft data consistency in k-space by linear mixing on observed locations.

    lam: 0..1, higher means stronger pull to measurements.
    """
    # Guard rails
    lam = float(lam)
    if lam <= 0.0:
        return x
    if lam >= 1.0:
        return dc_hard_kspace(x, k_us, mask)

    k_pred = fft2c(x)                                # [B,2,H,W]
    mask2 = mask.expand(-1, 2, -1, -1).to(k_pred)     # [B,2,H,W]
    # On observed k-space: k_new = (1-lam) k_pred + lam k_us
    k_new = k_pred * (1.0 - mask2) + ((1.0 - lam) * k_pred + lam * k_us) * mask2
    x_new = ifft2c(k_new)
    return x_new