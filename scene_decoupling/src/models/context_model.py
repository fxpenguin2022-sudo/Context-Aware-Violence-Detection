from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask import build_masks_by_mode
from .memvit_like import ResNetFeaturePyramid, StreamAttentionStack
from .memory_bank import StreamMemoryBundle


class ContextDecoupledMemoryModel(nn.Module):
    """
    Streamed context model with skeleton-guided dual-memory decoupling.

    This version keeps the same algorithmic structure as before but runs the
    streaming memory update in batch-vectorized form to reduce Python overhead.
    """

    def __init__(self, model_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.backbone = ResNetFeaturePyramid(model_cfg)

        self.proj_dim = int(model_cfg['proj_dim'])
        self.num_heads = int(model_cfg['num_heads'])
        self.attn_layers = int(model_cfg['attn_layers_per_stream'])
        self.mlp_ratio = float(model_cfg['mlp_ratio'])
        self.attn_dropout = float(model_cfg['attn_dropout'])
        self.proj_dropout = float(model_cfg['proj_dropout'])

        self.action_stream = StreamAttentionStack(
            embed_dim=self.proj_dim,
            num_heads=self.num_heads,
            num_layers=self.attn_layers,
            mlp_ratio=self.mlp_ratio,
            attn_dropout=self.attn_dropout,
            proj_dropout=self.proj_dropout,
        )
        self.scene_stream = StreamAttentionStack(
            embed_dim=self.proj_dim,
            num_heads=self.num_heads,
            num_layers=self.attn_layers,
            mlp_ratio=self.mlp_ratio,
            attn_dropout=self.attn_dropout,
            proj_dropout=self.proj_dropout,
        )

        mem_cfg = model_cfg['memory']
        self.action_len = int(mem_cfg['action_len'])
        self.scene_len = int(mem_cfg['scene_len'])
        self.action_pool = tuple(int(x) for x in mem_cfg['action_pool'])
        self.scene_pool = tuple(int(x) for x in mem_cfg['scene_pool'])
        self.stop_grad_memory = bool(mem_cfg.get('stop_grad_memory', True))
        self.compression_mode = str(mem_cfg.get('compression_mode', 'default')).lower()

        self.mask_mode = str(model_cfg.get('mask', {}).get('mode', 'skeleton')).lower()
        self.mask_align_strategy = str(model_cfg.get('mask', {}).get('align_strategy', 'max')).lower()
        self.mask_enforce_complement = bool(model_cfg.get('mask', {}).get('enforce_complement', False))
        if self.mask_align_strategy not in {'max', 'avg', 'conv'}:
            raise ValueError(f'Unsupported mask.align_strategy: {self.mask_align_strategy}')

        self._conv_scales = (1, 2, 4, 8, 16)
        self.mask_conv_fg_weight = nn.ParameterDict()
        self.mask_conv_fg_bias = nn.ParameterDict()
        self.mask_conv_bg_weight = nn.ParameterDict()
        self.mask_conv_bg_bias = nn.ParameterDict()
        if self.mask_align_strategy == 'conv':
            for s in self._conv_scales:
                key = f's{s}'
                init_w = torch.full((1, 1, s, s), 1.0 / float(s * s))
                self.mask_conv_fg_weight[key] = nn.Parameter(init_w.clone())
                self.mask_conv_fg_bias[key] = nn.Parameter(torch.zeros(1))
                self.mask_conv_bg_weight[key] = nn.Parameter(init_w.clone())
                self.mask_conv_bg_bias[key] = nn.Parameter(torch.zeros(1))

        self.decouple_mode = str(model_cfg.get('decouple', {}).get('mode', 'dual')).lower()

        fus_cfg = model_cfg['fusion']
        hidden_dim = int(fus_cfg['hidden_dim'])
        drop = float(fus_cfg['dropout'])
        self.fusion_head = nn.Sequential(
            nn.Linear(self.proj_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, self.proj_dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.classifier = nn.Linear(self.proj_dim, 1)

        self.mask_sigma = float(data_cfg.get('mask_sigma', 12.0))
        self.mask_threshold = float(data_cfg.get('mask_threshold', 0.05))
        self.frame_h = int(data_cfg['frame_size'][0])
        self.frame_w = int(data_cfg['frame_size'][1])

    def _resolve_pools(self) -> tuple[tuple[int, int], tuple[int, int]]:
        low = self.action_pool
        high = self.scene_pool
        mode = self.compression_mode
        if mode == 'default':
            return low, high
        if mode == 'all_high':
            return high, high
        if mode == 'all_low':
            return low, low
        if mode == 'reverse_mixed':
            return high, low
        raise ValueError(f'Unsupported compression_mode: {self.compression_mode}')

    def _pool_tokens(self, x: torch.Tensor, pool_hw: tuple[int, int]) -> torch.Tensor:
        # x: [B, L, C, H, W] -> [B, tokens, C]
        x = x.permute(0, 2, 1, 3, 4)
        x = F.adaptive_avg_pool3d(x, output_size=(1, pool_hw[0], pool_hw[1]))
        x = x.squeeze(2).flatten(2).transpose(1, 2)
        return x

    def _stack_tokens(
        self,
        feats: list[torch.Tensor],
        masks: list[torch.Tensor],
        pool_hw: tuple[int, int],
    ) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for feat, mask in zip(feats, masks):
            # feat: [B,L,C,H,W], mask: [B,L,1,H,W]
            chunks.append(self._pool_tokens(feat * mask, pool_hw))
        return torch.cat(chunks, dim=1)

    def _make_tokens(
        self,
        feats: list[torch.Tensor],
        fg_masks: list[torch.Tensor],
        bg_masks: list[torch.Tensor],
        action_pool: tuple[int, int],
        scene_pool: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.decouple_mode
        if mode == 'dual':
            action_tokens = self._stack_tokens(feats, fg_masks, action_pool)
            scene_tokens = self._stack_tokens(feats, bg_masks, scene_pool)
            return action_tokens, scene_tokens

        if mode == 'action_only':
            action_tokens = self._stack_tokens(feats, fg_masks, action_pool)
            scene_tokens = torch.zeros_like(action_tokens[:, :1])
            return action_tokens, scene_tokens

        if mode == 'scene_only':
            scene_tokens = self._stack_tokens(feats, bg_masks, scene_pool)
            action_tokens = torch.zeros_like(scene_tokens[:, :1])
            return action_tokens, scene_tokens

        if mode == 'mixed':
            full_masks = [torch.ones_like(m) for m in fg_masks]
            mixed_tokens = self._stack_tokens(feats, full_masks, action_pool)
            return mixed_tokens, mixed_tokens

        raise ValueError(f'Unsupported decouple mode: {mode}')

    @staticmethod
    def _masked_mean_over_time(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C], valid_mask: [B, T]
        w = valid_mask.to(dtype=x.dtype).unsqueeze(-1)
        num = (x * w).sum(dim=1)
        den = w.sum(dim=1).clamp_min(1.0)
        return num / den

    def _downsample_mask_learnable(
        self,
        x: torch.Tensor,
        target_hw: tuple[int, int],
        stream: str,
    ) -> torch.Tensor:
        # x: [B*L,1,H,W]
        th, tw = int(target_hw[0]), int(target_hw[1])
        ih, iw = int(x.shape[-2]), int(x.shape[-1])
        if ih == th and iw == tw:
            return x
        if ih % max(th, 1) != 0 or iw % max(tw, 1) != 0:
            return F.adaptive_avg_pool2d(x, output_size=(th, tw))

        sh = ih // max(th, 1)
        sw = iw // max(tw, 1)
        if sh != sw or sh not in self._conv_scales:
            return F.adaptive_avg_pool2d(x, output_size=(th, tw))

        key = f's{sh}'
        if stream == 'fg':
            w = self.mask_conv_fg_weight[key].to(dtype=x.dtype, device=x.device)
            b = self.mask_conv_fg_bias[key].to(dtype=x.dtype, device=x.device)
        else:
            w = self.mask_conv_bg_weight[key].to(dtype=x.dtype, device=x.device)
            b = self.mask_conv_bg_bias[key].to(dtype=x.dtype, device=x.device)

        out = F.conv2d(x, weight=w, bias=b, stride=(sh, sw), padding=0)
        if int(out.shape[-2]) != th or int(out.shape[-1]) != tw:
            out = F.interpolate(out, size=(th, tw), mode='bilinear', align_corners=False)
        return torch.sigmoid(out)

    def _build_mask_pyramid(
        self,
        fg_mask: torch.Tensor,
        bg_mask: torch.Tensor,
        stage_sizes: list[tuple[int, int]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # fg/bg: [B,L,1,H,W]
        if fg_mask.ndim != 5 or bg_mask.ndim != 5:
            raise ValueError('fg_mask/bg_mask must be [B,L,1,H,W]')

        b, l = int(fg_mask.shape[0]), int(fg_mask.shape[1])
        fg_flat = fg_mask.reshape(b * l, 1, int(fg_mask.shape[-2]), int(fg_mask.shape[-1]))
        bg_flat = bg_mask.reshape(b * l, 1, int(bg_mask.shape[-2]), int(bg_mask.shape[-1]))

        fg_levels: list[torch.Tensor] = []
        bg_levels: list[torch.Tensor] = []
        for h, w in stage_sizes:
            th, tw = int(h), int(w)
            if self.mask_align_strategy == 'max':
                fg_lvl = F.adaptive_max_pool2d(fg_flat, output_size=(th, tw))
                if self.mask_enforce_complement:
                    bg_lvl = (1.0 - fg_lvl).clamp(0.0, 1.0)
                else:
                    bg_lvl = F.adaptive_max_pool2d(bg_flat, output_size=(th, tw))
            elif self.mask_align_strategy == 'avg':
                fg_lvl = F.adaptive_avg_pool2d(fg_flat, output_size=(th, tw))
                if self.mask_enforce_complement:
                    bg_lvl = (1.0 - fg_lvl).clamp(0.0, 1.0)
                else:
                    bg_lvl = F.adaptive_avg_pool2d(bg_flat, output_size=(th, tw))
            else:  # conv
                fg_lvl = self._downsample_mask_learnable(fg_flat, (th, tw), stream='fg')
                if self.mask_enforce_complement:
                    bg_lvl = (1.0 - fg_lvl).clamp(0.0, 1.0)
                else:
                    bg_lvl = self._downsample_mask_learnable(bg_flat, (th, tw), stream='bg')

            fg_levels.append(fg_lvl.reshape(b, l, 1, th, tw))
            bg_levels.append(bg_lvl.reshape(b, l, 1, th, tw))

        return fg_levels, bg_levels

    def forward(
        self,
        clips: torch.Tensor,
        poses: torch.Tensor,
        clip_valid_mask: torch.Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        # clips: [B, Nc, L, 3, H, W], poses: [B, Nc, L, M, K, 3]
        b, nc, l, _, h, w = clips.shape
        if clip_valid_mask is None:
            clip_valid_mask = torch.ones((b, nc), device=clips.device, dtype=torch.bool)
        else:
            clip_valid_mask = clip_valid_mask.to(device=clips.device, dtype=torch.bool)

        streaming_backbone = bool(getattr(self.backbone, 'streaming_mode', False))
        shaped_levels: list[torch.Tensor] = []
        stage_sizes: list[tuple[int, int]] = []
        if streaming_backbone:
            # MeMViT online memory is stateful across calls: reset once per video batch.
            if hasattr(self.backbone, 'clear_memory'):
                self.backbone.clear_memory()
        elif bool(getattr(self.backbone, 'is_video_backbone', False)):
            # [B, Nc, L, 3, H, W] -> [B*Nc, 3, L, H, W]
            x = clips.reshape(b * nc, l, 3, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat_levels = self.backbone(x)
            for feat in feat_levels:
                # feat: [B*Nc, L, C, Hs, Ws]
                _, lt, c, hs, ws = feat.shape
                shaped_levels.append(feat.reshape(b, nc, lt, c, hs, ws))
                stage_sizes.append((hs, ws))
        else:
            x = clips.reshape(b * nc * l, 3, h, w)
            feat_levels = self.backbone(x)
            for feat in feat_levels:
                # feat: [B*Nc*L, C, Hs, Ws]
                _, c, hs, ws = feat.shape
                shaped_levels.append(feat.reshape(b, nc, l, c, hs, ws))
                stage_sizes.append((hs, ws))

        action_pool, scene_pool = self._resolve_pools()

        mem = StreamMemoryBundle(
            action_len=self.action_len,
            scene_len=self.scene_len,
            stop_grad=self.stop_grad_memory,
        )
        mem.reset()

        evol_steps: list[torch.Tensor] = []  # each [B,C]
        scene_steps: list[torch.Tensor] = []  # each [B,C]
        fg_ratio_steps: list[torch.Tensor] = []  # each [B]
        overlap_steps: list[torch.Tensor] = []  # each [B]
        valid_steps: list[torch.Tensor] = []  # each [B]

        for ti in range(nc):
            valid_t = clip_valid_mask[:, ti]
            if not bool(valid_t.any().item()):
                continue

            if streaming_backbone:
                cur_clip = clips[:, ti].permute(0, 2, 1, 3, 4).contiguous()  # [B,3,L,H,W]
                # Keep stable names for valid streams, and unique names for pads,
                # so padded slots do not accumulate stale online memory.
                video_names = [
                    (f'v{bi}' if bool(valid_t[bi].item()) else f'pad{bi}_{ti}')
                    for bi in range(b)
                ]
                cur_levels = self.backbone.forward_clip(cur_clip, video_names=video_names)
                if not stage_sizes:
                    stage_sizes = [(int(lvl.shape[-2]), int(lvl.shape[-1])) for lvl in cur_levels]
            else:
                cur_levels = [lvl[:, ti] for lvl in shaped_levels]  # each [B,L,C,H,W]
            cur_pose = poses[:, ti]  # [B,L,M,K,3]

            fg_mask, bg_mask = build_masks_by_mode(
                poses=cur_pose,
                mode=self.mask_mode,
                out_h=stage_sizes[0][0],
                out_w=stage_sizes[0][1],
                in_h=self.frame_h,
                in_w=self.frame_w,
                sigma=self.mask_sigma,
                threshold=self.mask_threshold,
            )
            fg_pyr, bg_pyr = self._build_mask_pyramid(fg_mask, bg_mask, stage_sizes)

            fg_ratio_steps.append(fg_mask.mean(dim=(1, 2, 3, 4)))
            level_overlaps = [(fg_lvl * bg_lvl).mean(dim=(1, 2, 3, 4)) for fg_lvl, bg_lvl in zip(fg_pyr, bg_pyr)]
            overlap_steps.append(torch.stack(level_overlaps, dim=1).mean(dim=1))
            valid_steps.append(valid_t)

            action_tokens, scene_tokens = self._make_tokens(
                feats=cur_levels,
                fg_masks=fg_pyr,
                bg_masks=bg_pyr,
                action_pool=action_pool,
                scene_pool=scene_pool,
            )

            q = cur_levels[-1].mean(dim=(1, 3, 4)).unsqueeze(1)  # [B,1,C]

            if self.decouple_mode in {'dual', 'action_only', 'mixed'}:
                action_kv, action_pad = mem.action.gather(action_tokens, valid_t)
                evo = self.action_stream(q, action_kv, key_padding_mask=action_pad)
                mem.action.append(action_tokens, valid_mask=valid_t)
            else:
                evo = torch.zeros_like(q)

            if self.decouple_mode in {'dual', 'scene_only', 'mixed'}:
                scene_kv, scene_pad = mem.scene.gather(scene_tokens, valid_t)
                sce = self.scene_stream(q, scene_kv, key_padding_mask=scene_pad)
                mem.scene.append(scene_tokens, valid_mask=valid_t)
            else:
                sce = torch.zeros_like(q)

            evol_steps.append(evo.squeeze(1))
            scene_steps.append(sce.squeeze(1))

        if evol_steps:
            evol_t = torch.stack(evol_steps, dim=1)      # [B,T,C]
            scene_t = torch.stack(scene_steps, dim=1)    # [B,T,C]
            fg_t = torch.stack(fg_ratio_steps, dim=1)    # [B,T]
            overlap_t = torch.stack(overlap_steps, dim=1)  # [B,T]
            valid_t = torch.stack(valid_steps, dim=1)    # [B,T]

            f_evol = self._masked_mean_over_time(evol_t, valid_t)
            f_scene = self._masked_mean_over_time(scene_t, valid_t)
            fg_ratio = self._masked_mean_over_time(fg_t.unsqueeze(-1), valid_t).squeeze(-1)
            mask_overlap = self._masked_mean_over_time(overlap_t.unsqueeze(-1), valid_t).squeeze(-1)
        else:
            f_evol = torch.zeros((b, self.proj_dim), dtype=clips.dtype, device=clips.device)
            f_scene = torch.zeros((b, self.proj_dim), dtype=clips.dtype, device=clips.device)
            fg_ratio = torch.zeros((b,), dtype=clips.dtype, device=clips.device)
            mask_overlap = torch.zeros((b,), dtype=clips.dtype, device=clips.device)
            evol_t = torch.zeros((b, 0, self.proj_dim), dtype=clips.dtype, device=clips.device)
            scene_t = torch.zeros((b, 0, self.proj_dim), dtype=clips.dtype, device=clips.device)
            fg_t = torch.zeros((b, 0), dtype=clips.dtype, device=clips.device)
            overlap_t = torch.zeros((b, 0), dtype=clips.dtype, device=clips.device)
            valid_t = torch.zeros((b, 0), dtype=torch.bool, device=clips.device)

        f_context = self.fusion_head(torch.cat([f_evol, f_scene], dim=-1))
        video_logit = self.classifier(f_context).squeeze(-1)
        video_prob = torch.sigmoid(video_logit)

        out = {
            'video_logit': video_logit,
            'video_prob': video_prob,
            'f_evol': f_evol,
            'f_scene': f_scene,
            'f_context': f_context,
            'video_feat': f_context,
            'fg_ratio': fg_ratio,
            'mask_overlap': mask_overlap,
        }
        if return_debug:
            out.update(
                {
                    'evol_steps': evol_t,
                    'scene_steps': scene_t,
                    'fg_ratio_steps': fg_t,
                    'mask_overlap_steps': overlap_t,
                    'valid_steps': valid_t,
                }
            )
        return out
