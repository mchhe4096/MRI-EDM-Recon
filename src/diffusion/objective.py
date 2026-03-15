import torch

def edm_sample_sigma(batch: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    """
    log-uniform sampling of sigma
    returns: [B,1,1,1]
    """
    u = torch.rand(batch, device=device)
    sigma = sigma_min * (sigma_max / sigma_min) ** u
    return sigma.view(batch, 1, 1, 1)


def edm_precond_coeffs(
    sigma: torch.Tensor,
    sigma_data: float,
    eps: float = 1e-12,
):
    """
    EDM preconditioning coefficients (Karras et al.):
      c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
      c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
      c_in   = 1 / sqrt(sigma^2 + sigma_data^2)
      c_noise = 0.25 * log(sigma)
    """
    sigma_data = float(sigma_data)
    sigma2 = sigma * sigma
    sd2 = sigma_data * sigma_data
    denom = torch.sqrt(sigma2 + sd2)

    c_skip = sd2 / (sigma2 + sd2)
    c_out = sigma * sigma_data / denom
    c_in = 1.0 / denom
    c_noise = 0.25 * torch.log(sigma + eps)
    return c_skip, c_out, c_in, c_noise


def edm_denoise(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    cond: torch.Tensor,
    sigma_data: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a preconditioned EDM denoiser.
    Model is assumed to output F_theta(c_in * x, sigma, cond); return x0 prediction.
    """
    c_skip, c_out, c_in, _ = edm_precond_coeffs(sigma, sigma_data=sigma_data)
    f_pred = model(c_in * x, sigma, cond)
    x0_pred = c_skip * x + c_out * f_pred
    return x0_pred, f_pred
