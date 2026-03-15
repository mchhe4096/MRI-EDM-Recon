import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class SigmaPosEmb(nn.Module):
    """
    Sinusoidal embedding of log-sigma.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, sigma_vec: torch.Tensor) -> torch.Tensor:
        # sigma_vec: [B,1]
        s = torch.log(sigma_vec.clamp_min(1e-12))
        half = self.dim // 2
        if half == 0:
            return s

        freqs = torch.arange(half, device=s.device, dtype=s.dtype)
        freqs = torch.exp(-math.log(10000.0) * freqs / max(half - 1, 1))
        angles = s * freqs.unsqueeze(0)  # [B, half]

        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, ch: int, temb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(_num_groups(ch), ch)
        self.gn2 = nn.GroupNorm(_num_groups(ch), ch)
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, ch * 2),
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.gn1(x)
        h = F.silu(h)

        # FiLM modulation from sigma/time embedding.
        scale_shift = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift

        h = self.conv1(h)
        h = self.conv2(F.silu(self.gn2(h)))
        return x + h


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNetV2(nn.Module):
    """
    forward(x, sigma, cond) -> model output for EDM objective.
    """

    def __init__(self, x_ch=2, cond_ch=3, base_ch=128, out_ch=2):
        super().__init__()
        in_ch = x_ch + cond_ch
        temb_dim = base_ch * 4

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.rb0 = ResBlock(base_ch, temb_dim)
        self.down1 = Down(base_ch, base_ch * 2)
        self.rb1 = ResBlock(base_ch * 2, temb_dim)

        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.rb2 = ResBlock(base_ch * 4, temb_dim)

        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.rb3 = ResBlock(base_ch * 8, temb_dim)

        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.rb_up2 = ResBlock(base_ch * 4, temb_dim)

        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.rb_up1 = ResBlock(base_ch * 2, temb_dim)

        self.up0 = Up(base_ch * 2, base_ch)
        self.rb_up0 = ResBlock(base_ch, temb_dim)

        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

        self.sigma_pos_emb = SigmaPosEmb(base_ch)
        self.sigma_mlp = nn.Sequential(
            nn.Linear(base_ch, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

    def _sigma_to_vec(self, sigma: torch.Tensor) -> torch.Tensor:
        # Accept [B,1,1,1], [B,1], or [B].
        if sigma.dim() == 4:
            s = sigma[:, :, 0, 0]
        elif sigma.dim() == 2:
            s = sigma
        elif sigma.dim() == 1:
            s = sigma.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported sigma shape: {tuple(sigma.shape)}")
        return s.view(-1, 1).float()

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        assert cond is not None
        xin = torch.cat([x, cond], dim=1)

        s = self._sigma_to_vec(sigma).to(xin.dtype)
        temb = self.sigma_mlp(self.sigma_pos_emb(s))

        h0 = self.in_conv(xin)
        h0 = self.rb0(h0, temb)

        h1 = self.down1(h0)
        h1 = self.rb1(h1, temb)

        h2 = self.down2(h1)
        h2 = self.rb2(h2, temb)

        h3 = self.down3(h2)
        h3 = self.rb3(h3, temb)

        u2 = self.up2(h3) + h2
        u2 = self.rb_up2(u2, temb)

        u1 = self.up1(u2) + h1
        u1 = self.rb_up1(u1, temb)

        u0 = self.up0(u1) + h0
        u0 = self.rb_up0(u0, temb)

        return self.out_conv(u0)
