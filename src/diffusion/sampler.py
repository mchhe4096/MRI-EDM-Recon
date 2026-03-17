import torch
from typing import Optional

from .objective import edm_denoise
from src.mri.data_consistency import dc_soft_kspace


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


def _init_state(
    shape,
    cond: torch.Tensor,
    sigma0: torch.Tensor,
    init_mode: str,
    init_blend: float,
) -> torch.Tensor:
    """
    Initialize sampling state for conditional reconstruction.

    init_mode:
      - "noise": pure Gaussian initialization (legacy behavior)
      - "zf": initialize from zero-filled reconstruction (cond[:, :2])
      - "blend": alpha * x_zf + (1-alpha) * noise, alpha=init_blend
    """
    device = cond.device
    noise = torch.randn(shape, device=device) * sigma0

    mode = str(init_mode).lower()
    if mode == "noise":
        return noise

    if cond.shape[1] < 2:
        raise ValueError(f"cond must have at least 2 channels for x_zf init, got shape={tuple(cond.shape)}")
    x_zf = cond[:, :2]
    if tuple(x_zf.shape) != tuple(shape):
        raise ValueError(f"x_zf shape mismatch: x_zf={tuple(x_zf.shape)} shape={tuple(shape)}")

    if mode == "zf":
        return x_zf.clone()

    if mode == "blend":
        alpha = float(max(0.0, min(1.0, init_blend)))
        return alpha * x_zf + (1.0 - alpha) * noise

    raise ValueError(f"Unsupported init_mode: {init_mode}")


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
    dc_start: float = 0.6,
    dc_every: int = 2,
    dc_lam: float = 0.15,
    dc_ramp: bool = False,
    sigma_data: float = 0.5,
    init_mode: str = "zf",
    init_blend: float = 0.5,
):
    """
    EDM-style sampler (Karras schedule + Euler / Heun).
    Assumes model predicts EDM F_theta(c_in * x, sigma, cond); converted to x0 internally.

    shape: (B,2,H,W)
    cond:  [B,C,H,W]
    k_us:  [B,2,H,W]
    mask:  [B,1,H,W]
    """
    device = next(model.parameters()).device
    B = shape[0]

    sigmas = _sigma_schedule_karras(steps, sigma_min, sigma_max, rho=rho, device=device)  # [steps+1]
    x = _init_state(
        shape=shape,
        cond=cond,
        sigma0=sigmas[0],
        init_mode=init_mode,
        init_blend=init_blend,
    )

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
        x0, _ = edm_denoise(model, x, sigma_b, cond, sigma_data=sigma_data)
        d = _to_d(x, x0, sigma_b)

        # Euler step
        x_euler = x + (sigma_next_b - sigma_b) * d

        if heun and sigma_next > 0:
            # Heun correction: evaluate derivative at next point
            x0_next, _ = edm_denoise(model, x_euler, sigma_next_b, cond, sigma_data=sigma_data)
            d_next = _to_d(x_euler, x0_next, sigma_next_b)
            x = x + (sigma_next_b - sigma_b) * (0.5 * d + 0.5 * d_next)
        else:
            x = x_euler

        # --- DC hook (late + low-frequency + soft) ---
        if dc:
            assert k_us is not None and mask is not None
            start_i = int(dc_start * steps)
            if (i >= start_i) and (dc_every > 0) and ((i - start_i) % dc_every == 0):

                if dc_ramp:
                    # linearly ramp lambda from dc_lam to 0.25 towards the end
                    end_i = steps - 1
                    denom = max(1, end_i - start_i)
                    t = (i - start_i) / denom
                    lam_i = (1.0 - t) * dc_lam + t * 0.25
                else:
                    lam_i = dc_lam

                x = dc_soft_kspace(x, k_us, mask, lam=lam_i)

    return x
