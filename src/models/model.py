from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import ViolenceSkateFormerLite
from .mil_head import MILHead


class ViolenceMILModel(nn.Module):
    def __init__(self, model_cfg: dict, data_cfg: dict) -> None:
        super().__init__()
        self.backbone = ViolenceSkateFormerLite(model_cfg=model_cfg, data_cfg=data_cfg)
        self.mil_head = MILHead(
            dim=int(model_cfg["d_model"]),
            topk_ratio=float(model_cfg["mil"]["topk_ratio"]),
            topk_beta=float(model_cfg["mil"]["topk_beta"]),
            attn_temperature=float(model_cfg["mil"].get("attn_temperature", 1.0)),
        )

    def forward(self, windows: torch.Tensor, window_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        # windows: [B, W, T, M, K, C]
        b, w, t, m, k, c = windows.shape

        x = windows.reshape(b * w, t, m, k, c)
        frame_valid = window_valid.reshape(b * w, t)
        _, window_feat = self.backbone(x, frame_valid=frame_valid)
        window_feat = window_feat.reshape(b, w, -1)

        bag_mask = window_valid.any(dim=-1)  # [B, W]
        out = self.mil_head(window_feat, window_mask=bag_mask)
        out["bag_mask"] = bag_mask
        return out

    def acg_state(self) -> dict[str, float]:
        if not getattr(self.backbone, "acg_enabled", False):
            return {"tau": 0.0, "temp": 0.0}
        return {
            "tau": float(self.backbone.acg.tau.detach().cpu().item()),
            "temp": float(self.backbone.acg.temperature().detach().cpu().item()),
        }
