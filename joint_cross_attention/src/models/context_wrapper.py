from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from scene_decoupling.src.common.checkpoint import load_checkpoint
from scene_decoupling.src.models.context_model import ContextDecoupledMemoryModel


def _resolve_checkpoint_path(path_str: str, branch_name: str) -> str:
    raw = str(path_str).strip()
    if not raw:
        raise ValueError(
            f"{branch_name} checkpoint is empty. Set the checkpoint path in the config or place the "
            f"expected file under ./checkpoints/."
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[3] / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{branch_name} checkpoint not found: {path}")
    return str(path)


class ContextBranchWrapper(nn.Module):
    def __init__(self, model_cfg: dict, data_cfg: dict, checkpoint_path: str) -> None:
        super().__init__()
        self.impl = ContextDecoupledMemoryModel(model_cfg=model_cfg, data_cfg=data_cfg)
        load_checkpoint(_resolve_checkpoint_path(checkpoint_path, 'scene_decoupling'), self.impl, map_location='cpu')
        if bool(model_cfg.get('backbone_act_checkpoint', False)):
            self.impl.backbone.enable_activation_checkpoint()
        for module_name in ['fusion_head', 'classifier']:
            module = getattr(self.impl, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, video_clips: torch.Tensor, video_poses: torch.Tensor, clip_valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.impl(video_clips, video_poses, clip_valid_mask=clip_valid_mask, return_debug=True)
        return {
            'evol_steps': out['evol_steps'],
            'scene_steps': out['scene_steps'],
            'valid_steps': out['valid_steps'],
            'f_evol': out['f_evol'],
            'f_scene': out['f_scene'],
            'fg_ratio': out['fg_ratio'],
            'mask_overlap': out['mask_overlap'],
        }
