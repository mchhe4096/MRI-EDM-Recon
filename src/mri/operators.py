import torch

def _to_complex(x2: torch.Tensor) -> torch.Tensor:
    # x2: [B,2,H,W] or [2,H,W]
    if x2.ndim == 3:
        x2 = x2.unsqueeze(0)
    return torch.complex(x2[:, 0], x2[:, 1])

def _to_2ch(xc: torch.Tensor) -> torch.Tensor:
    # xc: [B,H,W] complex -> [B,2,H,W]
    return torch.stack([xc.real, xc.imag], dim=1)

def fft2c(x2: torch.Tensor) -> torch.Tensor:
    """
    Centered FFT2.
    x2: [B,2,H,W] float -> k2: [B,2,H,W] float
    """
    xc = _to_complex(x2)
    xc = torch.fft.ifftshift(xc, dim=(-2, -1))
    kc = torch.fft.fft2(xc, dim=(-2, -1), norm="ortho")
    kc = torch.fft.fftshift(kc, dim=(-2, -1))
    return _to_2ch(kc)

def ifft2c(k2: torch.Tensor) -> torch.Tensor:
    """
    Centered IFFT2.
    k2: [B,2,H,W] float -> x2: [B,2,H,W] float
    """
    kc = _to_complex(k2)
    kc = torch.fft.ifftshift(kc, dim=(-2, -1))
    xc = torch.fft.ifft2(kc, dim=(-2, -1), norm="ortho")
    xc = torch.fft.fftshift(xc, dim=(-2, -1))
    return _to_2ch(xc)
