import torch
import torch.nn as nn

class SimpleUNetStub(nn.Module):
    def __init__(self, x_ch=2, cond_ch=3, out_ch=2, base_ch=64):
        super().__init__()
        in_ch = x_ch + cond_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
        )
        self.sigma_fc = nn.Linear(1, out_ch)

    def forward(self, x, sigma, cond=None):
        assert cond is not None
        xin = torch.cat([x, cond], dim=1)
        y = self.net(xin)
        log_sigma = sigma[:, :, 0, 0]
        b = self.sigma_fc(torch.log(log_sigma + 1e-12)).view(-1, y.shape[1], 1, 1)
        return y + b
