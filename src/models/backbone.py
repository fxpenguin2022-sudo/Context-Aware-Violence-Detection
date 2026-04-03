from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .acg import AdaptiveConfidenceGating
from .embeddings import SkeletonEmbedding
from .skateformer_block import SkateFormerLiteBlock


class ViolenceSkateFormerLite(nn.Module):
    def __init__(self, model_cfg: dict, data_cfg: dict) -> None:
        super().__init__()
        d_model = int(model_cfg["d_model"])
        depth = int(model_cfg["depth"])
        num_heads = int(model_cfg["num_heads"])
        mlp_ratio = float(model_cfg["mlp_ratio"])
        dropout = float(model_cfg["dropout"])
        attn_dropout = float(model_cfg["attn_dropout"])
        drop_path_max = float(model_cfg.get("drop_path", 0.0))

        self.use_grad_checkpoint = bool(model_cfg.get("use_grad_checkpoint", False))
        self.acg_enabled = bool(model_cfg["acg"].get("enabled", True))

        self.embedding = SkeletonEmbedding(
            input_dim=int(model_cfg["input_dim"]),
            d_model=d_model,
            max_persons=int(data_cfg["max_persons"]),
            num_keypoints=int(data_cfg["num_keypoints"]),
        )

        if self.acg_enabled:
            acg_cfg = model_cfg["acg"]
            self.acg = AdaptiveConfidenceGating(
                init_tau=float(acg_cfg["init_tau"]),
                init_temp=float(acg_cfg["init_temp"]),
                learnable_tau=bool(acg_cfg.get("learnable_tau", True)),
                learnable_temp=bool(acg_cfg.get("learnable_temp", True)),
            )

        dpr = torch.linspace(0, drop_path_max, depth).tolist()
        body_parts = model_cfg["interaction"]["body_parts"]
        interaction_enabled = bool(model_cfg["interaction"].get("enabled", True))
        self.blocks = nn.ModuleList(
            [
                SkateFormerLiteBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[i],
                    body_parts=body_parts,
                    interaction_enabled=interaction_enabled,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        windows: torch.Tensor,
        frame_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # windows: [N, T, M, K, C]
        conf = windows[..., 2:3]
        x = self.embedding(windows)

        if self.acg_enabled:
            gate = self.acg(conf)
            x = x * gate

        for block in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)

        if frame_valid is not None:
            weights = frame_valid.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [N, T, 1, 1, 1]
            pooled = (x * weights).sum(dim=(1, 2, 3))
            denom = weights.sum(dim=1).squeeze(-1).squeeze(-1).squeeze(-1)
            denom = torch.clamp(denom, min=1.0) * x.shape[2] * x.shape[3]
            pooled = pooled / denom.unsqueeze(-1)
        else:
            pooled = x.mean(dim=(1, 2, 3))

        return x, pooled
