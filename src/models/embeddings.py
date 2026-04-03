from __future__ import annotations

import math

import torch
import torch.nn as nn


class SkeletonEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int, max_persons: int, num_keypoints: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.person_embed = nn.Parameter(torch.zeros(max_persons, d_model))
        self.joint_embed = nn.Parameter(torch.zeros(num_keypoints, d_model))
        nn.init.trunc_normal_(self.person_embed, std=0.02)
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

    @staticmethod
    def _temporal_embedding(t: int, d: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe = torch.zeros(t, d, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, M, K, C]
        n, t, m, k, _ = x.shape
        out = self.input_proj(x)

        pe_t = self._temporal_embedding(t, out.shape[-1], out.device)[None, :, None, None, :]
        pe_p = self.person_embed[None, None, :m, None, :]
        pe_k = self.joint_embed[None, None, None, :k, :]
        out = out + pe_t + pe_p + pe_k
        return out
