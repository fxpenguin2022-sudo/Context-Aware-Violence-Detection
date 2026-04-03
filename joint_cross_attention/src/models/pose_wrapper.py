from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.model import ViolenceMILModel


def _resolve_checkpoint_path(path_str: str) -> str:
    raw = str(path_str).strip()
    if not raw:
        raise ValueError(
            "pose_branch checkpoint is empty. Set model.pose_branch.checkpoint or place the "
            "expected file under ./checkpoints/."
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[3] / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"pose_branch checkpoint not found: {path}")
    return str(path)


class PoseBranchWrapper(nn.Module):
    def __init__(
        self,
        model_cfg: dict,
        data_cfg: dict,
        checkpoint_path: str,
        strict_load: bool = True,
    ) -> None:
        super().__init__()
        self.impl = ViolenceMILModel(model_cfg=model_cfg, data_cfg=data_cfg)
        payload = torch.load(_resolve_checkpoint_path(checkpoint_path), map_location='cpu')
        self.impl.load_state_dict(payload['model'], strict=bool(strict_load))

    def forward(self, pose_windows: torch.Tensor, pose_window_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        b, w, t, m, k, c = pose_windows.shape
        x = pose_windows.reshape(b * w, t, m, k, c)
        frame_valid = pose_window_valid.reshape(b * w, t)
        _, window_feat = self.impl.backbone(x, frame_valid=frame_valid)
        window_feat = window_feat.reshape(b, w, -1)

        conf = pose_windows[..., 2].clamp(0.0, 1.0)
        frame_mask = pose_window_valid.to(dtype=pose_windows.dtype).unsqueeze(-1).unsqueeze(-1)
        conf_num = (conf * frame_mask).sum(dim=(1, 2, 3, 4))
        conf_den = frame_mask.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
        pose_conf_mean = conf_num / conf_den

        pose_valid_ratio = pose_window_valid.to(dtype=pose_windows.dtype).mean(dim=(1, 2))

        if getattr(self.impl.backbone, "acg_enabled", False):
            gate = self.impl.backbone.acg(conf.reshape(b * w, t, m, k, 1)).reshape(b, w, t, m, k)
            gate_num = (gate * frame_mask).sum(dim=(1, 2, 3, 4))
            gate_den = frame_mask.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
            acg_gate_mean = gate_num / gate_den
        else:
            acg_gate_mean = pose_conf_mean

        bag_mask = pose_window_valid.any(dim=-1)
        mil_out = self.impl.mil_head(window_feat, window_mask=bag_mask)
        window_probs = mil_out['window_probs']
        attn_weight = mil_out['attn_weight']

        topk = max(1, int(round(w * float(self.impl.mil_head.topk_ratio))))
        topk_idx = torch.topk(window_probs, k=topk, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, window_feat.shape[-1])
        topk_feat = torch.gather(window_feat, dim=1, index=gather_idx)
        topk_feat_mean = topk_feat.mean(dim=1)
        attn_feat = (attn_weight.unsqueeze(-1) * window_feat).sum(dim=1)
        beta = float(self.impl.mil_head.topk_beta)
        f_skel = beta * topk_feat_mean + (1.0 - beta) * attn_feat

        topk_mask = torch.gather(bag_mask, dim=1, index=topk_idx)
        topk_score_mean = torch.gather(window_probs, dim=1, index=topk_idx).mean(dim=1)
        skel_confidence = (0.45 * pose_conf_mean + 0.35 * acg_gate_mean + 0.20 * pose_valid_ratio).clamp(0.0, 1.0)
        return {
            'window_feat': window_feat,
            'window_probs': window_probs,
            'attn_weight': attn_weight,
            'bag_mask': bag_mask,
            'topk_idx': topk_idx,
            'topk_feat': topk_feat,
            'topk_mask': topk_mask,
            'f_skel': f_skel,
            'video_logit_pose': mil_out['video_logit'],
            'video_prob_pose': mil_out['video_prob'],
            'pose_conf_mean': pose_conf_mean,
            'acg_gate_mean': acg_gate_mean,
            'pose_valid_ratio': pose_valid_ratio,
            'topk_score_mean': topk_score_mean,
            'skel_confidence': skel_confidence,
        }
