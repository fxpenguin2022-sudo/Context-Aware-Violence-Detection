from __future__ import annotations

import torch
import torch.nn as nn


class CrossPersonInteraction(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, body_parts: list[list[int]]) -> None:
        super().__init__()
        self.body_parts = body_parts
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, M, K, D]
        n, t, m, k, d = x.shape
        out = torch.zeros_like(x)
        count = torch.zeros((n, t, m, k, 1), dtype=x.dtype, device=x.device)

        for joints in self.body_parts:
            valid = [j for j in joints if 0 <= j < k]
            if not valid:
                continue
            part = x[:, :, :, valid, :]  # [N, T, M, L, D]
            l = len(valid)
            part = part.reshape(n * t, m * l, d)
            part_out, _ = self.attn(part, part, part, need_weights=False)
            part_out = part_out.reshape(n, t, m, l, d)

            out[:, :, :, valid, :] += part_out
            count[:, :, :, valid, :] += 1.0

        # Joints outside body parts keep identity mapping to avoid losing information.
        count_safe = torch.clamp(count, min=1.0)
        out = out / count_safe
        out = torch.where(count == 0, x, out)
        return out
