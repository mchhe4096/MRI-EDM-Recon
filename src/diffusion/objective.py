import torch

def edm_sample_sigma(batch: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    """
    log-uniform sampling of sigma
    returns: [B,1,1,1]
    """
    u = torch.rand(batch, device=device)
    sigma = sigma_min * (sigma_max / sigma_min) ** u
    return sigma.view(batch, 1, 1, 1)
