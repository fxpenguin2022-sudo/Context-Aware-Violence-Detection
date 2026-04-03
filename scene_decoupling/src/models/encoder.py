from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, base_dim: int, out_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.GELU(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim * 2),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_dim * 2, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
