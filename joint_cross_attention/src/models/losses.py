from __future__ import annotations

import torch
import torch.nn as nn


class JointBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))
        self.label_smoothing = float(label_smoothing)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, video_logit: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.label_smoothing > 0:
            smooth = max(0.0, min(0.5, self.label_smoothing))
            label = label * (1.0 - smooth) + 0.5 * smooth
        loss = self.bce(video_logit, label)
        return {'loss': loss, 'loss_video': loss.detach()}
