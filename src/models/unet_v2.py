import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, ch)
        self.gn2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = self.conv1(F.silu(self.gn1(x)))
        h = self.conv2(F.silu(self.gn2(h)))
        return x + h


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNetV2(nn.Module):
    """
    Drop-in replacement for your SimpleUNetStub:
      forward(x, sigma, cond) -> x0_pred
    Strategy:
      - concat [x, cond]
      - UNet with residual blocks
      - add sigma-dependent bias (same idea as your current model)
    """
    def __init__(self, x_ch=2, cond_ch=3, base_ch=128, out_ch=2):
        super().__init__()
        in_ch = x_ch + cond_ch

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.rb0 = ResBlock(base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.rb1 = ResBlock(base_ch * 2)

        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.rb2 = ResBlock(base_ch * 4)

        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.rb3 = ResBlock(base_ch * 8)

        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.rb_up2 = ResBlock(base_ch * 4)

        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.rb_up1 = ResBlock(base_ch * 2)

        self.up0 = Up(base_ch * 2, base_ch)
        self.rb_up0 = ResBlock(base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

        # sigma bias: keep same spirit as your sigma_fc
        self.sigma_fc = nn.Sequential(
            nn.Linear(1, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, out_ch),
        )

    def forward(self, x, sigma, cond=None):
        assert cond is not None
        xin = torch.cat([x, cond], dim=1)

        h0 = self.in_conv(xin)
        h0 = self.rb0(h0)

        h1 = self.down1(h0)
        h1 = self.rb1(h1)

        h2 = self.down2(h1)
        h2 = self.rb2(h2)

        h3 = self.down3(h2)
        h3 = self.rb3(h3)

        u2 = self.up2(h3) + h2
        u2 = self.rb_up2(u2)

        u1 = self.up1(u2) + h1
        u1 = self.rb_up1(u1)

        u0 = self.up0(u1) + h0
        u0 = self.rb_up0(u0)

        y = self.out_conv(u0)

        # sigma: accept [B,1,1,1] or [B,1] variants
        if sigma.dim() == 4:
            s = sigma[:, :, 0, 0]
        else:
            s = sigma
        s = torch.log(s + 1e-12).view(-1, 1)  # [B,1]
        b = self.sigma_fc(s).view(-1, y.shape[1], 1, 1)

        return y + b
