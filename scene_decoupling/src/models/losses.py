from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 1.0,
        focal_gamma: float = 0.0,
        label_smoothing: float = 0.0,
        sep_weight: float = 0.0,
        overlap_weight: float = 0.0,
        fg_ratio_weight: float = 0.0,
        fg_ratio_min: float = 0.0,
        fg_ratio_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)
        self.sep_weight = float(sep_weight)
        self.overlap_weight = float(overlap_weight)
        self.fg_ratio_weight = float(fg_ratio_weight)
        self.fg_ratio_min = float(fg_ratio_min)
        self.fg_ratio_max = float(fg_ratio_max)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(
        self,
        video_logit: torch.Tensor,
        label: torch.Tensor,
        *,
        f_evol: torch.Tensor | None = None,
        f_scene: torch.Tensor | None = None,
        mask_overlap: torch.Tensor | None = None,
        fg_ratio: torch.Tensor | None = None,
        weight_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if self.label_smoothing > 0:
            smooth = max(0.0, min(0.5, self.label_smoothing))
            # Keep binary targets away from hard 0/1 to improve calibration.
            label = label * (1.0 - smooth) + 0.5 * smooth

        if self.focal_gamma <= 0:
            loss_video = self.bce(video_logit, label)
        else:
            p = torch.sigmoid(video_logit)
            pt = torch.where(label > 0.5, p, 1 - p)
            focal = (1 - pt).pow(self.focal_gamma)
            raw = nn.functional.binary_cross_entropy_with_logits(video_logit, label, reduction='none')
            loss_video = (focal * raw).mean()

        loss_sep = loss_video.new_zeros(())
        if self.sep_weight > 0 and f_evol is not None and f_scene is not None:
            evol_n = F.normalize(f_evol, dim=-1, eps=1e-6)
            scene_n = F.normalize(f_scene, dim=-1, eps=1e-6)
            cos = (evol_n * scene_n).sum(dim=-1)
            loss_sep = cos.pow(2).mean()

        loss_overlap = loss_video.new_zeros(())
        if self.overlap_weight > 0 and mask_overlap is not None:
            loss_overlap = mask_overlap.mean()

        loss_fg_ratio = loss_video.new_zeros(())
        if self.fg_ratio_weight > 0 and fg_ratio is not None:
            lower = torch.relu(self.fg_ratio_min - fg_ratio)
            upper = torch.relu(fg_ratio - self.fg_ratio_max)
            loss_fg_ratio = (lower + upper).mean()

        scale = max(0.0, float(weight_scale))
        loss = loss_video + scale * (
            self.sep_weight * loss_sep + self.overlap_weight * loss_overlap + self.fg_ratio_weight * loss_fg_ratio
        )

        return {
            'loss': loss,
            'loss_video': loss_video.detach(),
            'loss_sep': loss_sep.detach(),
            'loss_overlap': loss_overlap.detach(),
            'loss_fg_ratio': loss_fg_ratio.detach(),
            'constraint_scale': loss_video.new_tensor(scale).detach(),
        }
