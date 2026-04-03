from __future__ import annotations

import torch
import torch.nn as nn

from .interaction_partition import CrossPersonInteraction
from .utils import MLP, StochasticDepth


class SkateFormerLiteBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
        body_parts: list[list[int]],
        interaction_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.interaction_enabled = bool(interaction_enabled)

        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.interaction_attn = (
            CrossPersonInteraction(dim, num_heads, attn_dropout, body_parts) if self.interaction_enabled else None
        )

        self.dw_temporal = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim)
        self.dw_joint = nn.Conv3d(dim, dim, kernel_size=(1, 1, 3), padding=(0, 0, 1), groups=dim)

        self.fuse = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.drop_path = StochasticDepth(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def _temporal_branch(self, x: torch.Tensor) -> torch.Tensor:
        # [N, T, M, K, D] -> apply attention on T for each (M, K)
        n, t, m, k, d = x.shape
        y = x.permute(0, 2, 3, 1, 4).reshape(n * m * k, t, d)
        y, _ = self.temporal_attn(y, y, y, need_weights=False)
        y = y.reshape(n, m, k, t, d).permute(0, 3, 1, 2, 4)
        return y

    def _spatial_branch(self, x: torch.Tensor) -> torch.Tensor:
        # [N, T, M, K, D] -> apply attention on K for each (T, M)
        n, t, m, k, d = x.shape
        y = x.reshape(n * t * m, k, d)
        y, _ = self.spatial_attn(y, y, y, need_weights=False)
        y = y.reshape(n, t, m, k, d)
        return y

    def _conv_temporal_branch(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 4, 1, 2, 3)  # [N, D, T, M, K]
        y = self.dw_temporal(y)
        return y.permute(0, 2, 3, 4, 1)

    def _conv_joint_branch(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 4, 1, 2, 3)  # [N, D, T, M, K]
        y = self.dw_joint(y)
        return y.permute(0, 2, 3, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        b1 = self._temporal_branch(y)
        b2 = self._spatial_branch(y)
        if self.interaction_attn is None:
            # Ablation: remove cross-person interaction branch while keeping other branches unchanged.
            b3 = torch.zeros_like(y)
        else:
            b3 = self.interaction_attn(y)
        b4 = self._conv_temporal_branch(y)
        b5 = self._conv_joint_branch(y)

        fused = self.fuse(torch.cat([b1, b2, b3, b4, b5], dim=-1))
        x = x + self.drop_path(fused)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
