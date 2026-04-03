from __future__ import annotations

import torch
import torch.nn as nn


class MILHead(nn.Module):
    def __init__(self, dim: int, topk_ratio: float, topk_beta: float, attn_temperature: float = 1.0) -> None:
        super().__init__()
        self.window_classifier = nn.Linear(dim, 1)
        self.attn_proj = nn.Linear(dim, 1)
        self.topk_ratio = float(topk_ratio)
        self.topk_beta = float(topk_beta)
        self.attn_temperature = float(attn_temperature)

    @staticmethod
    def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(min=eps, max=1.0 - eps)
        return torch.log(p / (1.0 - p))

    def forward(self, window_feat: torch.Tensor, window_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # window_feat: [B, W, D]
        logits = self.window_classifier(window_feat).squeeze(-1)  # [B, W]
        probs = torch.sigmoid(logits)

        if window_mask is not None:
            mask = window_mask.float()
            probs = probs * mask
        else:
            mask = torch.ones_like(probs)

        w = probs.shape[1]
        topk = max(1, int(round(w * self.topk_ratio)))
        topk_vals, _ = torch.topk(probs, k=topk, dim=1)
        topk_score = topk_vals.mean(dim=1)

        attn_logits = self.attn_proj(window_feat).squeeze(-1) / self.attn_temperature
        attn_logits = attn_logits.masked_fill(mask == 0, -1e4)
        attn_weight = torch.softmax(attn_logits, dim=1)
        attn_score = (attn_weight * probs).sum(dim=1)

        video_prob = self.topk_beta * topk_score + (1.0 - self.topk_beta) * attn_score
        video_logit = self._safe_logit(video_prob)

        return {
            "window_logits": logits,
            "window_probs": probs,
            "video_prob": video_prob,
            "video_logit": video_logit,
            "attn_weight": attn_weight,
        }
