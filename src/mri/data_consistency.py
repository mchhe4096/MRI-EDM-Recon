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