import torch

def cartesian_mask(shape_hw, accel: int = 4, center_frac: float = 0.08, device=None) -> torch.Tensor:
    """
    Returns mask: [1,1,H,W] float {0,1}
    - Fully sample a low-frequency band in the center (width = center_frac * W)
    - Sample remaining columns with step = accel (simple uniform undersampling)
    """
    H, W = shape_hw
    m = torch.zeros(W, device=device)

    # center band
    c = int(round(W * center_frac))
    c = max(2, c)
    start = (W - c) // 2
    m[start:start + c] = 1.0

    # uniform sampling outside center
    m[::accel] = 1.0

    mask = m.view(1, 1, 1, W).repeat(1, 1, H, 1)  # [1,1,H,W]
    return mask
