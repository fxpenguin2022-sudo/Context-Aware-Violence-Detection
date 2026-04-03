from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from joint_cross_attention.src.models.context_wrapper import ContextBranchWrapper
from joint_cross_attention.src.models.pose_wrapper import PoseBranchWrapper


class FeedForwardEnhancer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ffn(x))


class JointViolenceModel(nn.Module):
    def __init__(self, model_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> None:
        super().__init__()
        pose_cfg = model_cfg['pose_branch']
        context_cfg = model_cfg['context_branch']
        self.pose_branch = PoseBranchWrapper(
            model_cfg=pose_cfg['model_cfg'],
            data_cfg=data_cfg['pose_branch'],
            checkpoint_path=str(pose_cfg['checkpoint']),
            strict_load=bool(pose_cfg.get('strict_load', True)),
        )
        self.context_branch = ContextBranchWrapper(
            model_cfg=context_cfg['model_cfg'],
            data_cfg=data_cfg['context_branch'],
            checkpoint_path=str(context_cfg['checkpoint']),
        )

        sg_cfg = model_cfg.get('sg_aca', {})
        amcf_cfg = model_cfg.get('amcf', {})
        cls_cfg = model_cfg.get('classifier', {})
        ctx_fusion_cfg = model_cfg.get('context_fusion', {})

        d_joint = int(sg_cfg['d_joint'])
        pose_dim = int(pose_cfg['model_cfg']['d_model'])
        context_dim = int(context_cfg['model_cfg']['proj_dim'])
        heads = int(sg_cfg['num_heads'])
        attn_drop = float(sg_cfg.get('dropout', 0.1))
        self.sg_aca_enabled = bool(sg_cfg.get('enabled', True))
        self.sg_aca_mode = 'disabled' if not self.sg_aca_enabled else str(sg_cfg.get('mode', 'asymmetric')).lower()
        if self.sg_aca_mode not in {'disabled', 'asymmetric', 'symmetric'}:
            raise ValueError(f'Unsupported sg_aca.mode: {self.sg_aca_mode}')
        self.context_stream_mode = str(ctx_fusion_cfg.get('mode', 'dual')).lower()
        if self.context_stream_mode not in {'dual', 'single'}:
            raise ValueError(f'Unsupported context_fusion.mode: {self.context_stream_mode}')
        if self.context_stream_mode != 'dual' and self.sg_aca_mode != 'disabled':
            raise ValueError('SG-ACA requires context_fusion.mode=dual.')
        self.single_context_source = str(ctx_fusion_cfg.get('single_source', 'mean')).lower()
        if self.single_context_source not in {'mean', 'evol', 'scene'}:
            raise ValueError(f'Unsupported context_fusion.single_source: {self.single_context_source}')

        self.amcf_mode = str(amcf_cfg.get('mode', 'learned')).lower()
        if self.amcf_mode not in {'learned', 'fixed_avg', 'concat', 'static_learned'}:
            raise ValueError(f'Unsupported amcf.mode: {self.amcf_mode}')
        alpha_guidance_cfg = amcf_cfg.get('alpha_guidance', {})
        self.alpha_guidance_enabled = bool(alpha_guidance_cfg.get('enabled', False))
        self.alpha_guidance_scale = float(alpha_guidance_cfg.get('scale', 2.0))
        self.alpha_guidance_center = float(alpha_guidance_cfg.get('center', 0.60))
        self.alpha_guidance_sharpness = float(alpha_guidance_cfg.get('sharpness', 10.0))
        self.alpha_guidance_detach = bool(alpha_guidance_cfg.get('detach_reliability', True))
        if self.alpha_guidance_enabled and self.amcf_mode != 'learned':
            raise ValueError('amcf.alpha_guidance requires amcf.mode=learned.')

        self.d_joint = d_joint
        self.num_modalities = 3 if self.context_stream_mode == 'dual' else 2
        self.fusion_dim = d_joint * self.num_modalities
        self.skel_res_proj = nn.Linear(pose_dim, d_joint)
        self.evol_direct_proj = nn.Linear(context_dim, d_joint)
        self.scene_direct_proj = nn.Linear(context_dim, d_joint)
        self.sg_dropout = nn.Dropout(attn_drop)

        if self.sg_aca_mode in {'asymmetric', 'symmetric'}:
            self.skel_q_proj = nn.Linear(pose_dim, d_joint)
            self.evol_k_proj = nn.Linear(context_dim, d_joint)
            self.evol_v_proj = nn.Linear(context_dim, d_joint)
            self.scene_k_proj = nn.Linear(context_dim, d_joint)
            self.scene_v_proj = nn.Linear(context_dim, d_joint)
            self.attn_evol = nn.MultiheadAttention(embed_dim=d_joint, num_heads=heads, dropout=attn_drop, batch_first=True)
            self.attn_scene = nn.MultiheadAttention(embed_dim=d_joint, num_heads=heads, dropout=attn_drop, batch_first=True)
            self.evol_norm = nn.LayerNorm(d_joint)
            self.scene_norm = nn.LayerNorm(d_joint)

        if self.sg_aca_mode == 'symmetric':
            self.skel_k_proj = nn.Linear(pose_dim, d_joint)
            self.skel_v_proj = nn.Linear(pose_dim, d_joint)
            self.evol_q_proj = nn.Linear(context_dim, d_joint)
            self.scene_q_proj = nn.Linear(context_dim, d_joint)
            self.attn_evol_rev = nn.MultiheadAttention(embed_dim=d_joint, num_heads=heads, dropout=attn_drop, batch_first=True)
            self.attn_scene_rev = nn.MultiheadAttention(embed_dim=d_joint, num_heads=heads, dropout=attn_drop, batch_first=True)

        if self.amcf_mode == 'learned':
            reduction = int(amcf_cfg['reduction'])
            gate_hidden = max(1, self.fusion_dim // reduction)
            self.gate = nn.Sequential(
                nn.Linear(self.fusion_dim, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, self.num_modalities),
            )
        elif self.amcf_mode == 'static_learned':
            self.static_gate_logits = nn.Parameter(torch.zeros(self.num_modalities, dtype=torch.float32))

        if self.amcf_mode == 'concat':
            concat_hidden = int(amcf_cfg.get('concat_ffn_hidden_dim', max(int(amcf_cfg.get('ffn_hidden_dim', 1024)) * self.num_modalities, self.fusion_dim)))
            self.concat_enhancer = FeedForwardEnhancer(
                dim=self.fusion_dim,
                hidden_dim=concat_hidden,
                dropout=float(amcf_cfg.get('dropout', 0.2)),
            )
            self.concat_classifier = nn.Sequential(
                nn.Dropout(float(cls_cfg.get('dropout', 0.2))),
                nn.Linear(self.fusion_dim, 1),
            )
        else:
            self.enhancer = FeedForwardEnhancer(
                dim=d_joint,
                hidden_dim=int(amcf_cfg.get('ffn_hidden_dim', 1024)),
                dropout=float(amcf_cfg.get('dropout', 0.2)),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(float(cls_cfg.get('dropout', 0.2))),
                nn.Linear(d_joint, 1),
            )

    def _align_asymmetric(
        self,
        skel_query: torch.Tensor,
        skel_residual: torch.Tensor,
        f_evol_raw: torch.Tensor,
        f_scene_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        evol_k = self.evol_k_proj(f_evol_raw).unsqueeze(1)
        evol_v = self.evol_v_proj(f_evol_raw).unsqueeze(1)
        scene_k = self.scene_k_proj(f_scene_raw).unsqueeze(1)
        scene_v = self.scene_v_proj(f_scene_raw).unsqueeze(1)

        aligned_evol_vec, _ = self.attn_evol(query=skel_query, key=evol_k, value=evol_v, need_weights=False)
        aligned_scene_vec, _ = self.attn_scene(query=skel_query, key=scene_k, value=scene_v, need_weights=False)

        f_evol_aligned = self.evol_norm(skel_residual + self.sg_dropout(aligned_evol_vec.squeeze(1)))
        f_scene_aligned = self.scene_norm(skel_residual + self.sg_dropout(aligned_scene_vec.squeeze(1)))
        return f_evol_aligned, f_scene_aligned

    def _align_symmetric(
        self,
        f_skel_raw: torch.Tensor,
        skel_query: torch.Tensor,
        skel_residual: torch.Tensor,
        f_evol_raw: torch.Tensor,
        f_scene_raw: torch.Tensor,
        evol_residual: torch.Tensor,
        scene_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skel_k = self.skel_k_proj(f_skel_raw).unsqueeze(1)
        skel_v = self.skel_v_proj(f_skel_raw).unsqueeze(1)

        evol_q = self.evol_q_proj(f_evol_raw).unsqueeze(1)
        evol_k = self.evol_k_proj(f_evol_raw).unsqueeze(1)
        evol_v = self.evol_v_proj(f_evol_raw).unsqueeze(1)

        scene_q = self.scene_q_proj(f_scene_raw).unsqueeze(1)
        scene_k = self.scene_k_proj(f_scene_raw).unsqueeze(1)
        scene_v = self.scene_v_proj(f_scene_raw).unsqueeze(1)

        skel_from_evol, _ = self.attn_evol(query=skel_query, key=evol_k, value=evol_v, need_weights=False)
        evol_from_skel, _ = self.attn_evol_rev(query=evol_q, key=skel_k, value=skel_v, need_weights=False)
        skel_from_scene, _ = self.attn_scene(query=skel_query, key=scene_k, value=scene_v, need_weights=False)
        scene_from_skel, _ = self.attn_scene_rev(query=scene_q, key=skel_k, value=skel_v, need_weights=False)

        f_evol_aligned = self.evol_norm(
            0.5 * (skel_residual + self.sg_dropout(skel_from_evol.squeeze(1)))
            + 0.5 * (evol_residual + self.sg_dropout(evol_from_skel.squeeze(1)))
        )
        f_scene_aligned = self.scene_norm(
            0.5 * (skel_residual + self.sg_dropout(skel_from_scene.squeeze(1)))
            + 0.5 * (scene_residual + self.sg_dropout(scene_from_skel.squeeze(1)))
        )
        return f_evol_aligned, f_scene_aligned

    def _resolve_amcf_weights(
        self,
        fusion_in: torch.Tensor,
        skel_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = fusion_in.shape[0]
        alpha_guidance_bias = torch.zeros((batch,), dtype=fusion_in.dtype, device=fusion_in.device)
        if self.amcf_mode == 'learned':
            gate_logits = self.gate(fusion_in)
            if self.alpha_guidance_enabled and skel_confidence is not None:
                rel = skel_confidence.detach() if self.alpha_guidance_detach else skel_confidence
                rel = rel.to(dtype=fusion_in.dtype, device=fusion_in.device)
                alpha_guidance_bias = torch.sigmoid((rel - self.alpha_guidance_center) * self.alpha_guidance_sharpness)
                gate_logits[:, 0] = gate_logits[:, 0] + self.alpha_guidance_scale * alpha_guidance_bias
            weights = torch.softmax(gate_logits, dim=-1)
        elif self.amcf_mode == 'fixed_avg':
            weights = torch.full((batch, 3), 1.0 / 3.0, dtype=fusion_in.dtype, device=fusion_in.device)
        elif self.amcf_mode == 'static_learned':
            shared = torch.softmax(self.static_gate_logits.to(device=fusion_in.device, dtype=fusion_in.dtype), dim=-1)
            weights = shared.unsqueeze(0).expand(batch, -1)
        elif self.amcf_mode == 'concat':
            weights = torch.full((batch, self.num_modalities), 1.0 / float(self.num_modalities), dtype=fusion_in.dtype, device=fusion_in.device)
        else:
            raise RuntimeError(f'Unhandled amcf.mode: {self.amcf_mode}')
        alpha = weights[:, 0]
        beta = weights[:, 1]
        gamma = weights[:, 2] if self.num_modalities == 3 else torch.zeros_like(alpha)
        return alpha, beta, gamma, alpha_guidance_bias

    def _select_single_context(
        self,
        evol_residual: torch.Tensor,
        scene_residual: torch.Tensor,
    ) -> torch.Tensor:
        if self.single_context_source == 'evol':
            return evol_residual
        if self.single_context_source == 'scene':
            return scene_residual
        return 0.5 * (evol_residual + scene_residual)

    def forward(
        self,
        pose_windows: torch.Tensor,
        pose_window_valid: torch.Tensor,
        video_clips: torch.Tensor,
        video_poses: torch.Tensor,
        clip_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pose_out = self.pose_branch(pose_windows, pose_window_valid)
        context_out = self.context_branch(video_clips, video_poses, clip_valid_mask)

        f_skel_raw = pose_out['f_skel']
        f_evol_raw = context_out['f_evol']
        f_scene_raw = context_out['f_scene']
        skel_confidence = pose_out['skel_confidence']

        f_skel_proj = self.skel_res_proj(f_skel_raw)
        evol_residual = self.evol_direct_proj(f_evol_raw)
        scene_residual = self.scene_direct_proj(f_scene_raw)

        if self.context_stream_mode == 'single':
            f_context_single = self._select_single_context(evol_residual=evol_residual, scene_residual=scene_residual)
            f_evol_aligned = f_context_single
            f_scene_aligned = torch.zeros_like(f_context_single)
            active_features = [f_skel_proj, f_context_single]
        else:
            if self.sg_aca_mode == 'disabled':
                f_evol_aligned = evol_residual
                f_scene_aligned = scene_residual
            else:
                skel_query = self.skel_q_proj(f_skel_raw).unsqueeze(1)
                if self.sg_aca_mode == 'asymmetric':
                    f_evol_aligned, f_scene_aligned = self._align_asymmetric(
                        skel_query=skel_query,
                        skel_residual=f_skel_proj,
                        f_evol_raw=f_evol_raw,
                        f_scene_raw=f_scene_raw,
                    )
                else:
                    f_evol_aligned, f_scene_aligned = self._align_symmetric(
                        f_skel_raw=f_skel_raw,
                        skel_query=skel_query,
                        skel_residual=f_skel_proj,
                        f_evol_raw=f_evol_raw,
                        f_scene_raw=f_scene_raw,
                        evol_residual=evol_residual,
                        scene_residual=scene_residual,
                    )
            active_features = [f_skel_proj, f_evol_aligned, f_scene_aligned]

        fusion_in = torch.cat(active_features, dim=-1)
        alpha, beta, gamma, alpha_guidance_bias = self._resolve_amcf_weights(
            fusion_in=fusion_in,
            skel_confidence=skel_confidence,
        )

        if self.amcf_mode == 'concat':
            f_fused_raw = fusion_in
            f_fused = self.concat_enhancer(f_fused_raw)
            video_logit = self.concat_classifier(f_fused).squeeze(-1)
        else:
            if self.context_stream_mode == 'single':
                f_fused_raw = (
                    alpha.unsqueeze(-1) * f_skel_proj
                    + beta.unsqueeze(-1) * f_evol_aligned
                )
            else:
                f_fused_raw = (
                    alpha.unsqueeze(-1) * f_skel_proj
                    + beta.unsqueeze(-1) * f_evol_aligned
                    + gamma.unsqueeze(-1) * f_scene_aligned
                )
            f_fused = self.enhancer(f_fused_raw)
            video_logit = self.classifier(f_fused).squeeze(-1)
        video_prob = torch.sigmoid(video_logit)

        return {
            'video_logit': video_logit,
            'video_prob': video_prob,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'alpha_guidance_bias': alpha_guidance_bias,
            'f_skel': f_skel_raw,
            'f_skel_proj': f_skel_proj,
            'skel_confidence': skel_confidence,
            'pose_conf_mean': pose_out['pose_conf_mean'],
            'acg_gate_mean': pose_out['acg_gate_mean'],
            'pose_valid_ratio': pose_out['pose_valid_ratio'],
            'topk_score_mean': pose_out['topk_score_mean'],
            'f_evol_raw': f_evol_raw,
            'f_scene_raw': f_scene_raw,
            'f_evol_aligned': f_evol_aligned,
            'f_scene_aligned': f_scene_aligned,
            'f_fused_raw': f_fused_raw,
            'f_fused': f_fused,
            'window_probs': pose_out['window_probs'],
            'attn_weight': pose_out['attn_weight'],
            'bag_mask': pose_out['bag_mask'],
            'topk_feat': pose_out['topk_feat'],
            'topk_mask': pose_out['topk_mask'],
            'fg_ratio': context_out['fg_ratio'],
            'mask_overlap': context_out['mask_overlap'],
        }
