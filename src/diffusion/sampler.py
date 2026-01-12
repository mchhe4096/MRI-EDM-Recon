import torch
from src.mri.data_consistency import dc_hard_kspace

@torch.no_grad()
def sample_with_optional_dc(model, shape, cond, steps, sigma_min, sigma_max,
                            dc=False, k_us=None, mask=None, step_size=0.1):
    device = next(model.parameters()).device
    B = shape[0]
    x = torch.randn(shape, device=device) * sigma_max

    sigmas = torch.exp(torch.linspace(
        torch.log(torch.tensor(sigma_max, device=device)),
        torch.log(torch.tensor(sigma_min, device=device)),
        steps,
        device=device
    ))

    for i in range(steps):
        sigma = sigmas[i].view(1,1,1,1).expand(B,1,1,1)
        x0_pred = model(x, sigma, cond)

        # placeholder update（先跑通闭环；后面你们再换 EDM / DPM-Solver）
        x = x + (x0_pred - x) * step_size

        # DC hook
        if dc:
            assert k_us is not None and mask is not None
            x = dc_hard_kspace(x, k_us, mask)

    return x
