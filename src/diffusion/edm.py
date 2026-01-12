import torch
import torch.nn.functional as F

from .objective import edm_sample_sigma
from .sampler import sample_with_optional_dc


class EDMTrainer:
    """
    Minimal EDM-like wrapper:
      - loss: sample sigma, add noise, predict x0, MSE(x0_pred, x0)
      - sample: call sampler (optionally with DC)
    """
    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def loss(self, model, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x0:   [B,2,H,W]
        cond: [B,C,H,W]
        """
        device = x0.device
        B = x0.shape[0]

        # sample sigma per-sample
        sigma = edm_sample_sigma(B, self.sigma_min, self.sigma_max, device=device)  # [B]
        sigma_img = sigma.view(B, 1, 1, 1)

        n = torch.randn_like(x0)
        x_noisy = x0 + sigma_img * n

        # model predicts x0 directly (simple + stable for now)
        x0_pred = model(x_noisy, sigma_img, cond)

        return F.mse_loss(x0_pred, x0)

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        cond: torch.Tensor,
        steps: int = 40,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        dc: bool = False,
        k_us: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        shape: (B,2,H,W)
        cond:  [B,C,H,W]
        """
        return sample_with_optional_dc(
            model=model,
            shape=shape,
            cond=cond,
            steps=steps,
            sigma_min=float(self.sigma_min if sigma_min is None else sigma_min),
            sigma_max=float(self.sigma_max if sigma_max is None else sigma_max),
            dc=dc,
            k_us=k_us,
            mask=mask,
        )
