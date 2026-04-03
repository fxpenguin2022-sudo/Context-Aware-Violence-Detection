from __future__ import annotations

import torch
import torch.nn as nn


class VideoBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, focal_gamma: float = 0.0, window_sparsity_weight: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.focal_gamma = float(focal_gamma)
        self.window_sparsity_weight = float(window_sparsity_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(
        self,
        video_logit: torch.Tensor,
        label: torch.Tensor,
        window_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base_loss = self.bce(video_logit, label)

        if self.focal_gamma > 0:
            p = torch.sigmoid(video_logit)
            pt = torch.where(label > 0.5, p, 1 - p)
            focal = (1 - pt).pow(self.focal_gamma)
            base_loss = (focal * nn.functional.binary_cross_entropy_with_logits(video_logit, label, reduction="none")).mean()

        reg = torch.tensor(0.0, device=video_logit.device)
        if self.window_sparsity_weight > 0 and window_probs is not None:
            reg = window_probs.mean() * self.window_sparsity_weight

        total = base_loss + reg
        return {
            "loss": total,
            "loss_video": base_loss.detach(),
            "loss_reg": reg.detach(),
        }
