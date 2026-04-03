from __future__ import annotations

import importlib
import json
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import yaml


def _build_resnet50(pretrained: bool) -> nn.Module:
    if not pretrained:
        return tvm.resnet50(weights=None)

    # Try pretrained weights, but fall back to random init when offline.
    candidates = []
    if hasattr(tvm, 'ResNet50_Weights'):
        weights_enum = tvm.ResNet50_Weights
        if hasattr(weights_enum, 'IMAGENET1K_V2'):
            candidates.append(weights_enum.IMAGENET1K_V2)
        if hasattr(weights_enum, 'IMAGENET1K_V1'):
            candidates.append(weights_enum.IMAGENET1K_V1)

    for w in candidates:
        try:
            return tvm.resnet50(weights=w)
        except Exception:
            continue
    return tvm.resnet50(weights=None)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state = payload
    if isinstance(payload, dict):
        for key in ('model_state', 'state_dict', 'model', 'model_state_dict'):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f'Unsupported checkpoint payload type: {type(state)}')

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        k = str(key)
        if k.startswith('module.'):
            k = k[len('module.') :]
        if k.startswith('model.'):
            k = k[len('model.') :]
        if k.startswith('backbone.'):
            k = k[len('backbone.') :]
        cleaned[k] = value
    return cleaned


def _tokens_to_feature_map(tokens: torch.Tensor, thw: tuple[int, int, int], has_cls: bool) -> torch.Tensor:
    if has_cls:
        tokens = tokens[:, 1:, :]
    b, n, c = tokens.shape
    t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
    expected = t * h * w
    if n != expected:
        raise ValueError(f'Invalid token length for THW={thw}: got {n}, expect {expected}')
    x = tokens.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
    return x


def _install_memvit_import_shims() -> None:
    """
    Install lightweight runtime shims so the upstream MeMViT code can be imported
    in minimal environments without adding heavy dependencies.
    """

    if 'simplejson' not in sys.modules:
        m = types.ModuleType('simplejson')
        m.dumps = lambda obj, **kwargs: json.dumps(obj, sort_keys=kwargs.get('sort_keys', False), default=float)
        sys.modules['simplejson'] = m

    if 'iopath.common.file_io' not in sys.modules:
        iopath_mod = types.ModuleType('iopath')
        common_mod = types.ModuleType('iopath.common')
        file_io_mod = types.ModuleType('iopath.common.file_io')

        class _PathMgr:
            def open(self, *args, **kwargs):
                return open(*args, **kwargs)

            def exists(self, path: str) -> bool:
                return os.path.exists(path)

            def ls(self, path: str):
                return os.listdir(path)

            def mkdirs(self, path: str) -> None:
                os.makedirs(path, exist_ok=True)

        file_io_mod.g_pathmgr = _PathMgr()
        common_mod.file_io = file_io_mod
        iopath_mod.common = common_mod
        sys.modules['iopath'] = iopath_mod
        sys.modules['iopath.common'] = common_mod
        sys.modules['iopath.common.file_io'] = file_io_mod

    if 'fvcore.common.registry' not in sys.modules:
        fv_mod = types.ModuleType('fvcore')
        fv_common_mod = types.ModuleType('fvcore.common')
        fv_registry_mod = types.ModuleType('fvcore.common.registry')

        class Registry:
            def __init__(self, name: str):
                self._name = name
                self._obj: dict[str, Any] = {}

            def register(self):
                def _deco(obj):
                    self._obj[obj.__name__] = obj
                    return obj

                return _deco

            def get(self, name: str):
                return self._obj[name]

        fv_registry_mod.Registry = Registry
        fv_common_mod.registry = fv_registry_mod

        fv_nn_mod = types.ModuleType('fvcore.nn')
        fv_wi_mod = types.ModuleType('fvcore.nn.weight_init')

        def c2_msra_fill(module: nn.Module) -> None:
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        fv_wi_mod.c2_msra_fill = c2_msra_fill
        fv_nn_mod.weight_init = fv_wi_mod

        fv_mod.common = fv_common_mod
        fv_mod.nn = fv_nn_mod
        sys.modules['fvcore'] = fv_mod
        sys.modules['fvcore.common'] = fv_common_mod
        sys.modules['fvcore.common.registry'] = fv_registry_mod
        sys.modules['fvcore.nn'] = fv_nn_mod
        sys.modules['fvcore.nn.weight_init'] = fv_wi_mod

    if 'detectron2.layers' not in sys.modules:
        d2_mod = types.ModuleType('detectron2')
        d2_layers_mod = types.ModuleType('detectron2.layers')

        class ROIAlign(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()

            def forward(self, *args, **kwargs):
                raise RuntimeError('ROIAlign shim should not be used when DETECTION.ENABLE=False')

        d2_layers_mod.ROIAlign = ROIAlign
        d2_mod.layers = d2_layers_mod
        sys.modules['detectron2'] = d2_mod
        sys.modules['detectron2.layers'] = d2_layers_mod


class _CfgNode:
    """Tiny attribute-style config node used to feed official MeMViT builders."""

    def __init__(self, data: dict[str, Any]) -> None:
        for k, v in data.items():
            setattr(self, k, self._convert(v))

    def _convert(self, value: Any) -> Any:
        if isinstance(value, dict):
            return _CfgNode(value)
        if isinstance(value, list):
            return [self._convert(x) for x in value]
        return value


def _apply_memvit_cfg_defaults(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg_dict)
    if 'MEMVIT' not in out and 'ONLINE_MEM' in out:
        out['MEMVIT'] = out['ONLINE_MEM']

    out.setdefault('MODEL', {})
    out['MODEL'].setdefault('ACT_CHECKPOINT', False)
    out['MODEL'].setdefault('NUM_CLASSES', 400)
    out['MODEL'].setdefault('NUM_CLASSES_LIST', [])

    out.setdefault('MVIT', {})
    out['MVIT'].setdefault('SEPARATE_QKV', False)
    out['MVIT'].setdefault('POOL_KV_STRIDE', None)
    out['MVIT'].setdefault('NORM_STEM', False)
    out['MVIT'].setdefault('ZERO_DECAY_POS_CLS', False)
    out['MVIT'].setdefault('BOX_DEPTH', 0)
    out['MVIT'].setdefault('BOX_DROPOUT_RATE', 0.0)
    out['MVIT'].setdefault('BOX_DROPPATH_RATE', 0.0)
    out['MVIT'].setdefault('BOX_DROP_ATTN_RATE', 0.0)
    out['MVIT'].setdefault('BOX_DROP_QKV_RATE', 0.0)
    out['MVIT'].setdefault('FRAME_LEVEL', False)

    out.setdefault('DETECTION', {})
    out['DETECTION']['ENABLE'] = False

    out.setdefault('MEMVIT', {})
    out['MEMVIT'].setdefault('ENABLE', True)
    out['MEMVIT'].setdefault('ATTN_MAX_LEN', 3)
    out['MEMVIT'].setdefault('EXCLUDE_LAYERS', [])
    out['MEMVIT'].setdefault('COMPRESS', {'ENABLE': False, 'POOL_KERNEL': [7, 3, 3], 'POOL_STRIDE': [4, 2, 2]})
    sampler = str(out['MEMVIT'].get('SAMPLER', 'all'))
    # Current upstream sample_memory() supports only `all` or `gapN`.
    if sampler != 'all' and ('gap' not in sampler):
        out['MEMVIT']['SAMPLER'] = 'all'

    out['NUM_GPUS'] = 0
    return out


def _default_memvit_pretrain_path() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    cands = []
    env_path = os.environ.get('VIOLENCE_MEMVIT_CKPT', '').strip()
    if env_path:
        cands.append(env_path)
    cands.extend(
        [
            str(repo_root / 'checkpoints' / 'MeMViT_16L_16x4_K400.pyth'),
            str(repo_root / 'weights' / 'MeMViT_16L_16x4_K400.pyth'),
        ]
    )
    for p in cands:
        if os.path.exists(p):
            return p
    return ''


def _resolve_repo_path(path_str: str, fallback: Path) -> str:
    raw = str(path_str).strip()
    path = Path(os.path.expanduser(raw)) if raw else fallback
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[3] / path).resolve()
    return str(path)


class _ResNetFeaturePyramidImpl(nn.Module):
    """ResNet backbone with projected multi-scale features (1/8, 1/16, 1/32)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        name = str(cfg.get('backbone', 'resnet50_imagenet_pretrained')).lower()
        pretrained = bool('pretrained' in name)
        proj_dim = int(cfg['proj_dim'])
        freeze_stages = int(cfg.get('backbone_freeze_stages', 1))

        base = _build_resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.proj_s8 = nn.Conv2d(512, proj_dim, kernel_size=1)
        self.proj_s16 = nn.Conv2d(1024, proj_dim, kernel_size=1)
        self.proj_s32 = nn.Conv2d(2048, proj_dim, kernel_size=1)

        self._freeze_stages(freeze_stages)

    def _freeze_module(self, module: nn.Module) -> None:
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def _freeze_stages(self, freeze_stages: int) -> None:
        if freeze_stages >= 1:
            self._freeze_module(self.stem)
            self._freeze_module(self.layer1)
        if freeze_stages >= 2:
            self._freeze_module(self.layer2)
        if freeze_stages >= 3:
            self._freeze_module(self.layer3)
        if freeze_stages >= 4:
            self._freeze_module(self.layer4)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        s8 = self.layer2(x)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)

        p8 = self.proj_s8(s8)
        p16 = self.proj_s16(s16)
        p32 = self.proj_s32(s32)
        return [p8, p16, p32]


class _MemViTFeaturePyramidImpl(nn.Module):
    """
    Paper-style modified MemViT backbone.

    We keep the dual-stream decoupled head in the scene-decoupling branch and swap the feature encoder
    with a K400-pretrained MViT/MeMViT-style trunk. Checkpoint loading is
    always done with strict=False to support structure mismatch in the head.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        _install_memvit_import_shims()

        repo_path = _resolve_repo_path(
            str(cfg.get('memvit_repo_path', '')).strip(),
            Path('third_party') / 'MeMViT',
        )
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f'MeMViT repo not found: {repo_path}')
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        module = importlib.import_module('memvit.models.video_model_builder')
        mvit_cls = getattr(module, 'MViT')

        pretrained_path = str(cfg.get('backbone_pretrained_path', '')).strip()
        if not pretrained_path:
            pretrained_path = _default_memvit_pretrain_path()
        if not pretrained_path:
            raise FileNotFoundError(
                'No MeMViT checkpoint path provided. Put MeMViT_16L_16x4_K400.pyth under '
                'checkpoints/ or set model.backbone_pretrained_path / VIOLENCE_MEMVIT_CKPT.'
            )

        ckpt_path = _resolve_repo_path(pretrained_path, Path('checkpoints') / 'MeMViT_16L_16x4_K400.pyth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Configured backbone_pretrained_path does not exist: {pretrained_path}')

        try:
            payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            payload = torch.load(ckpt_path, map_location='cpu')
        raw_cfg = payload.get('cfg', None)
        if isinstance(raw_cfg, str):
            cfg_dict = yaml.safe_load(raw_cfg)
        elif isinstance(raw_cfg, dict):
            cfg_dict = raw_cfg
        else:
            raise TypeError(f'Unsupported cfg payload inside checkpoint: {type(raw_cfg)}')

        cfg_dict = _apply_memvit_cfg_defaults(cfg_dict)
        self.backbone_act_checkpoint = bool(cfg.get('backbone_act_checkpoint', False))
        self._act_checkpoint_applied = False
        if self.backbone_act_checkpoint:
            cfg_dict.setdefault('MODEL', {})
            # Always instantiate/load the plain modules first.
            # We enable local activation recomputation only after the outer
            # model checkpoint has been loaded, otherwise strict key matching
            # breaks on wrapped submodule names.
            cfg_dict['MODEL']['ACT_CHECKPOINT'] = False
        self.memvit_cfg = _CfgNode(cfg_dict)

        self.mvit = mvit_cls(self.memvit_cfg)
        state_dict = _extract_state_dict(payload)
        missing, unexpected = self.mvit.load_state_dict(state_dict, strict=False)

        loaded = len(state_dict) - len(unexpected)
        warnings.warn(
            (
                f'MemViT strict=False load from {ckpt_path}: '
                f'loaded={loaded} missing={len(missing)} unexpected={len(unexpected)}'
            ),
            stacklevel=2,
        )

        proj_dim = int(cfg['proj_dim'])
        self.input_temporal = int(cfg.get('memvit_input_temporal', int(self.memvit_cfg.DATA.NUM_FRAMES)))
        self.capture_blocks = tuple(int(x) for x in cfg.get('memvit_capture_blocks', [1, 3, 14]))
        if len(self.capture_blocks) != 3:
            raise ValueError(f'memvit_capture_blocks must contain 3 stages, got {self.capture_blocks}')

        cap_dims: list[int] = []
        for blk_idx in self.capture_blocks:
            if blk_idx < 0 or blk_idx >= len(self.mvit.blocks):
                raise ValueError(f'memvit_capture_blocks out of range: {self.capture_blocks}')
            cap_dims.append(int(self.mvit.blocks[blk_idx].norm2.normalized_shape[0]))

        self.proj_0 = nn.Conv3d(cap_dims[0], proj_dim, kernel_size=1)
        self.proj_1 = nn.Conv3d(cap_dims[1], proj_dim, kernel_size=1)
        self.proj_2 = nn.Conv3d(cap_dims[2], proj_dim, kernel_size=1)

        freeze_stages = int(cfg.get('backbone_freeze_stages', 0))
        self._freeze_stages(freeze_stages)
        self.has_cls_token = bool(self.mvit.cls_embed_on)

    def _freeze_module(self, module: nn.Module) -> None:
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def _freeze_stages(self, freeze_stages: int) -> None:
        if freeze_stages <= 0:
            return

        self._freeze_module(self.mvit.patch_embed)
        if hasattr(self.mvit, 'norm_stem') and self.mvit.norm_stem is not None:
            self._freeze_module(self.mvit.norm_stem)

        total = len(self.mvit.blocks)
        # Freeze evenly by depth fraction so this stays valid across variants.
        k = min(total, max(0, int(round(total * min(max(freeze_stages / 4.0, 0.0), 1.0)))))
        for blk in self.mvit.blocks[:k]:
            self._freeze_module(blk)

        if freeze_stages >= 4:
            self._freeze_module(self.mvit.norm)

    def _apply_activation_checkpoint(self) -> None:
        import torch.utils.checkpoint as torch_checkpoint

        class _CheckpointWrapper(nn.Module):
            def __init__(self, wrapped: nn.Module) -> None:
                super().__init__()
                self.wrapped = wrapped

            def __getattr__(self, name: str):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    wrapped = super().__getattr__('wrapped')
                    return getattr(wrapped, name)

            def forward(self, *args, **kwargs):
                if not torch.is_grad_enabled():
                    return self.wrapped(*args, **kwargs)

                def _run(*inner_args):
                    return self.wrapped(*inner_args, **kwargs)

                return torch_checkpoint.checkpoint(_run, *args, use_reentrant=False)

        self.mvit.patch_embed = _CheckpointWrapper(self.mvit.patch_embed)
        self.mvit.blocks = nn.ModuleList([_CheckpointWrapper(blk) for blk in self.mvit.blocks])
        self._act_checkpoint_applied = True

    def enable_activation_checkpoint(self) -> None:
        if self.backbone_act_checkpoint and not self._act_checkpoint_applied:
            self._apply_activation_checkpoint()

    def _project_level(self, tokens: torch.Tensor, thw: tuple[int, int, int], proj: nn.Conv3d, out_t: int) -> torch.Tensor:
        x = _tokens_to_feature_map(tokens, thw, has_cls=self.has_cls_token)  # [B,C,T,H,W]
        x = proj(x)
        if x.shape[2] != out_t:
            x = F.interpolate(x, size=(out_t, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
        return x.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]

    def clear_memory(self) -> None:
        if hasattr(self.mvit, 'clear_memory'):
            self.mvit.clear_memory()

    @staticmethod
    def _conv_out_dim(size: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
        # Matches Conv{2,3}d output-size formula with floor division.
        return ((int(size) + 2 * int(padding) - int(dilation) * (int(kernel) - 1) - 1) // int(stride)) + 1

    def forward_clip(self, x: torch.Tensor, video_names: list[str]) -> list[torch.Tensor]:
        # x: [B,3,T,H,W] for one streaming clip step.
        if x.ndim != 5:
            raise ValueError(f'MemViT backbone expects 5D input [B,3,T,H,W], got {tuple(x.shape)}')

        out_t = int(x.shape[2])
        if out_t != self.input_temporal:
            x = F.interpolate(
                x,
                size=(self.input_temporal, int(x.shape[3]), int(x.shape[4])),
                mode='trilinear',
                align_corners=False,
            )

        # Follow official MeMViT forward() up to encoder outputs.
        in_t, in_h, in_w = int(x.shape[2]), int(x.shape[3]), int(x.shape[4])
        pt, ph, pw = (int(v) for v in self.mvit.patch_stride)
        k_t, k_h, k_w = (int(v) for v in self.memvit_cfg.MVIT.PATCH_KERNEL)
        p_t, p_h, p_w = (int(v) for v in self.memvit_cfg.MVIT.PATCH_PADDING)
        t = self._conv_out_dim(in_t, k_t, pt, p_t)
        h = self._conv_out_dim(in_h, k_h, ph, p_h)
        w = self._conv_out_dim(in_w, k_w, pw, p_w)
        x = self.mvit.patch_embed(x)
        # Keep THW consistent with actual tokenized shape if runtime sizes differ.
        n = int(x.shape[1])
        if n != t * h * w:
            base = max(1, h * w)
            t = max(1, n // base)
            rem = max(1, n // t)
            if rem % max(1, h) == 0:
                w = rem // max(1, h)
            else:
                h = max(1, int(round(rem ** 0.5)))
                w = max(1, rem // h)
        b = int(x.shape[0])

        if bool(self.mvit.cls_embed_on):
            cls_tokens = self.mvit.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if bool(self.mvit.use_abs_pos):
            if bool(self.mvit.sep_pos_embed):
                pos_embed = self.mvit.pos_embed_spatial.repeat(1, self.mvit.patch_dims[0], 1) + torch.repeat_interleave(
                    self.mvit.pos_embed_temporal,
                    self.mvit.patch_dims[1] * self.mvit.patch_dims[2],
                    dim=1,
                )
                if bool(self.mvit.cls_embed_on):
                    pos_embed = torch.cat([self.mvit.pos_embed_class, pos_embed], 1)
                x = x + pos_embed
            else:
                x = x + self.mvit.pos_embed

        if float(self.mvit.drop_rate) > 0:
            x = self.mvit.pos_drop(x)

        if self.mvit.norm_stem is not None:
            x = self.mvit.norm_stem(x)

        mem_selections = self.mvit.sample_memory() if bool(self.mvit.use_online_memory) else None
        thw = [t, h, w]

        captured: dict[int, tuple[torch.Tensor, tuple[int, int, int]]] = {}
        for idx, block in enumerate(self.mvit.blocks):
            cur_selection = [] if idx in self.memvit_cfg.MEMVIT.EXCLUDE_LAYERS else mem_selections
            x, thw = block(x, thw, cur_selection, video_names)
            if idx in self.capture_blocks:
                captured[idx] = (x, (int(thw[0]), int(thw[1]), int(thw[2])))

        missing = [i for i in self.capture_blocks if i not in captured]
        if missing:
            raise RuntimeError(f'Missing MemViT capture blocks: {missing}')

        l0 = self._project_level(captured[self.capture_blocks[0]][0], captured[self.capture_blocks[0]][1], self.proj_0, out_t)
        l1 = self._project_level(captured[self.capture_blocks[1]][0], captured[self.capture_blocks[1]][1], self.proj_1, out_t)
        l2 = self._project_level(captured[self.capture_blocks[2]][0], captured[self.capture_blocks[2]][1], self.proj_2, out_t)
        return [l0, l1, l2]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Stateless one-shot fallback: clear memory and run a single clip batch.
        self.clear_memory()
        b = int(x.shape[0])
        names = [f'clip_{i}' for i in range(b)]
        return self.forward_clip(x, video_names=names)


class ResNetFeaturePyramid(nn.Module):
    """
    Unified backbone entry.

    - ResNet mode (legacy): input [N,3,H,W], outputs [N,C,H,W]x3
    - MemViT mode: input [N,3,T,H,W], outputs [N,T,C,H,W]x3
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        name = str(cfg.get('backbone', 'resnet50_imagenet_pretrained')).lower()
        self.is_video_backbone = 'memvit' in name or 'mvit' in name
        self.streaming_mode = False
        if self.is_video_backbone:
            self.impl = _MemViTFeaturePyramidImpl(cfg)
            self.streaming_mode = True
        else:
            self.impl = _ResNetFeaturePyramidImpl(cfg)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.impl(x)

    def clear_memory(self) -> None:
        if hasattr(self.impl, 'clear_memory'):
            self.impl.clear_memory()

    def forward_clip(self, x: torch.Tensor, video_names: list[str]) -> list[torch.Tensor]:
        if not hasattr(self.impl, 'forward_clip'):
            raise RuntimeError('forward_clip is only available for streaming backbones.')
        return self.impl.forward_clip(x, video_names=video_names)

    def enable_activation_checkpoint(self) -> None:
        if hasattr(self.impl, 'enable_activation_checkpoint'):
            self.impl.enable_activation_checkpoint()


class StreamAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        attn_dropout: float,
        proj_dropout: float,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(proj_dropout)
        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(proj_dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(
            qn,
            kvn,
            kvn,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = q + self.drop(attn_out)
        x = x + self.ffn(x)
        return x


class StreamAttentionStack(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        attn_dropout: float,
        proj_dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                StreamAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = q
        for block in self.blocks:
            x = block(x, kv, key_padding_mask)
        return x
