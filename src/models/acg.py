from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveConfidenceGating(nn.Module):
    def __init__(
        self,
        init_tau: float = 0.35,
        init_temp: float = 0.10,
        learnable_tau: bool = True,
        learnable_temp: bool = True,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.eps = eps

        tau = torch.tensor(float(init_tau), dtype=torch.float32)
        temp = torch.tensor(float(init_temp), dtype=torch.float32)

        self.tau = nn.Parameter(tau, requires_grad=learnable_tau)
        # Optimize temperature in unconstrained space and map with softplus.
        self.temp_raw = nn.Parameter(torch.log(torch.exp(temp) - 1.0), requires_grad=learnable_temp)

    def temperature(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.temp_raw) + self.eps

    def forward(self, conf: torch.Tensor) -> torch.Tensor:
        # conf: [N, T, M, K, 1]
        gate = torch.sigmoid((conf - self.tau) / self.temperature())
        return gate
