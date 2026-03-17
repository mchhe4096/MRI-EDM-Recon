import torch
import torch.nn.functional as F

from .objective import edm_precond_coeffs, edm_sample_sigma
from .sampler import sample_with_optional_dc


class EDMTrainer:
    """
    EDM wrapper:
      - loss: sample sigma, preconditioned denoiser objective
      - sample: call sampler (optionally with DC)
    """
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
    ):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)

    def loss(self, model, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x0:   [B,2,H,W]
        cond: [B,C,H,W]
        """
        device = x0.device
        B = x0.shape[0]

        # sample sigma per-sample
        sigma = edm_sample_sigma(B, self.sigma_min, self.sigma_max, device=device)

        n = torch.randn_like(x0)
        x_noisy = x0 + sigma * n

        # Standard EDM preconditioning.
        # This is equivalent to weighted x0-MSE with w(sigma)=1/c_out^2.
        c_skip, c_out, c_in, _ = edm_precond_coeffs(sigma, sigma_data=self.sigma_data)
        f_pred = model(c_in * x_noisy, sigma, cond)
        f_target = (x0 - c_skip * x_noisy) / (c_out + 1e-12)
        return F.mse_loss(f_pred, f_target)

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

        # --- new: soft-DC schedule knobs ---
        dc_start: float = 0.6,
        dc_every: int = 2,
        dc_lam: float = 0.15,
        dc_ramp: bool = False,

        sigma_data: float | None = None,
        init_mode: str = "zf",
        init_blend: float = 0.5,
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

            # --- new: pass-through to sampler ---
            dc_start=dc_start,
            dc_every=dc_every,
            dc_lam=dc_lam,
            dc_ramp=dc_ramp,

            sigma_data=float(self.sigma_data if sigma_data is None else sigma_data),
            init_mode=init_mode,
            init_blend=init_blend,
        )
