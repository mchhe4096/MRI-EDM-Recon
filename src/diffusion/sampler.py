import torch
from typing import Optional

from src.mri.data_consistency import dc_hard_kspace


def _sigma_schedule_karras(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device,
):
    """
    Karras noise schedule used by EDM:
      sigma(t) = (sigma_max^(1/rho) + t*(sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
    returns sigmas: [steps+1], descending, with last element = 0
    """
    i = torch.linspace(0, 1, steps, device=device)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max ** inv_rho + i * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])  # append 0 for final
    return sigmas


def _to_d(x: torch.Tensor, x0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Convert x0 prediction into EDM ODE derivative:
      d = (x - x0) / sigma
    sigma: [B,1,1,1]
    """
    return (x - x0) / (sigma + 1e-12)


@torch.no_grad()
def sample_with_optional_dc(
    model,
    shape,
    cond: torch.Tensor,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    dc: bool = False,
    k_us: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    rho: float = 7.0,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    heun: bool = True,
):
    """
    EDM-style sampler (Karras schedule + Euler / Heun).
    Assumes model predicts x0 given (x, sigma, cond).

    shape: (B,2,H,W)
    cond:  [B,C,H,W]
    k_us:  [B,2,H,W]
    mask:  [B,1,H,W]
    """
    device = next(model.parameters()).device
    B = shape[0]

    sigmas = _sigma_schedule_karras(steps, sigma_min, sigma_max, rho=rho, device=device)  # [steps+1]
    # initial noise
    x = torch.randn(shape, device=device) * sigmas[0]

    # optional: "churn" (stochasticity) like EDM paper; default off (s_churn=0)
    for i in range(steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # expand to [B,1,1,1]
        sigma_b = sigma.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        sigma_next_b = sigma_next.view(1, 1, 1, 1).expand(B, 1, 1, 1)

        # --- churn (optional) ---
        if s_churn > 0 and (s_tmin <= float(sigma) <= s_tmax):
            gamma = min(s_churn / steps, 2 ** 0.5 - 1)
        else:
            gamma = 0.0

        if gamma > 0:
            sigma_hat = sigma * (1 + gamma)
            sigma_hat_b = sigma_hat.view(1, 1, 1, 1).expand(B, 1, 1, 1)
            eps = torch.randn_like(x) * s_noise
            x = x + (sigma_hat_b**2 - sigma_b**2).sqrt() * eps
            sigma_b = sigma_hat_b  # use sigma_hat for denoise step

        # --- denoise (predict x0) ---
        x0 = model(x, sigma_b, cond)
        d = _to_d(x, x0, sigma_b)

        # Euler step
        x_euler = x + (sigma_next_b - sigma_b) * d

        if heun and sigma_next > 0:
            # Heun correction: evaluate derivative at next point
            x0_next = model(x_euler, sigma_next_b, cond)
            d_next = _to_d(x_euler, x0_next, sigma_next_b)
            x = x + (sigma_next_b - sigma_b) * (0.5 * d + 0.5 * d_next)
        else:
            x = x_euler

        # --- DC hook after each step (recommended) ---
        if dc:
            assert k_us is not None and mask is not None
            x = dc_hard_kspace(x, k_us, mask)

    return x
