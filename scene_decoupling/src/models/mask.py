from __future__ import annotations

import torch
import torch.nn.functional as F


def _skeleton_heatmap(
    poses: torch.Tensor,
    out_h: int,
    out_w: int,
    in_h: int,
    in_w: int,
    sigma: float,
) -> torch.Tensor:
    """
    poses: [B, L, M, K, 3] with x,y,conf in input-frame coordinates.
    returns heatmap in [B, L, H, W].
    """
    b, l, m, k, _ = poses.shape
    device = poses.device

    pts = poses[..., :2].reshape(b, l, m * k, 2)
    conf = poses[..., 2].reshape(b, l, m * k)

    scale_x = float(out_w) / float(max(in_w, 1))
    scale_y = float(out_h) / float(max(in_h, 1))
    pts_x = pts[..., 0] * scale_x
    pts_y = pts[..., 1] * scale_y

    ys = torch.arange(out_h, device=device, dtype=poses.dtype)
    xs = torch.arange(out_w, device=device, dtype=poses.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    gx = gx.view(1, 1, 1, out_h, out_w)
    gy = gy.view(1, 1, 1, out_h, out_w)

    px = pts_x.unsqueeze(-1).unsqueeze(-1)
    py = pts_y.unsqueeze(-1).unsqueeze(-1)
    pc = conf.unsqueeze(-1).unsqueeze(-1)

    sigma_feat = max(1e-6, sigma * (scale_x + scale_y) * 0.5)
    dist2 = (gx - px).pow(2) + (gy - py).pow(2)
    heat = torch.exp(-dist2 / (2.0 * sigma_feat * sigma_feat)) * pc
    return heat.max(dim=2).values


def build_skeleton_masks(
    poses: torch.Tensor,
    out_h: int,
    out_w: int,
    in_h: int,
    in_w: int,
    sigma: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """returns fg/bg masks in [B, L, 1, out_h, out_w]."""
    heat = _skeleton_heatmap(
        poses=poses,
        out_h=out_h,
        out_w=out_w,
        in_h=in_h,
        in_w=in_w,
        sigma=sigma,
    )
    fg = (heat > threshold).to(dtype=poses.dtype).unsqueeze(2)
    bg = 1.0 - fg
    return fg, bg


def build_masks_by_mode(
    poses: torch.Tensor,
    mode: str,
    out_h: int,
    out_w: int,
    in_h: int,
    in_w: int,
    sigma: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(mode).lower()
    if mode == 'skeleton':
        return build_skeleton_masks(
            poses=poses,
            out_h=out_h,
            out_w=out_w,
            in_h=in_h,
            in_w=in_w,
            sigma=sigma,
            threshold=threshold,
        )

    if mode == 'none':
        b, l = poses.shape[:2]
        fg = torch.ones((b, l, 1, out_h, out_w), dtype=poses.dtype, device=poses.device)
        bg = torch.ones_like(fg)
        return fg, bg

    if mode == 'random':
        sk_fg, _ = build_skeleton_masks(
            poses=poses,
            out_h=out_h,
            out_w=out_w,
            in_h=in_h,
            in_w=in_w,
            sigma=sigma,
            threshold=threshold,
        )
        # Match foreground occupancy ratio per frame.
        ratio = sk_fg.mean(dim=(2, 3, 4), keepdim=True)
        rand = torch.rand_like(sk_fg)
        fg = (rand < ratio).to(dtype=poses.dtype)
        bg = 1.0 - fg
        return fg, bg

    raise ValueError(f'Unsupported mask mode: {mode}')


def build_multiscale_mask_pyramid(
    fg_mask: torch.Tensor,
    bg_mask: torch.Tensor,
    stage_sizes: list[tuple[int, int]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build max-pooled masks aligned to feature pyramid stages."""
    if fg_mask.ndim != 5 or bg_mask.ndim != 5:
        raise ValueError('fg_mask/bg_mask must be [B,L,1,H,W]')

    b, l = fg_mask.shape[:2]
    fg_flat = fg_mask.reshape(b * l, 1, fg_mask.shape[-2], fg_mask.shape[-1])
    bg_flat = bg_mask.reshape(b * l, 1, bg_mask.shape[-2], bg_mask.shape[-1])

    fg_levels: list[torch.Tensor] = []
    bg_levels: list[torch.Tensor] = []
    for h, w in stage_sizes:
        fg_lvl = F.adaptive_max_pool2d(fg_flat, output_size=(h, w)).reshape(b, l, 1, h, w)
        bg_lvl = F.adaptive_max_pool2d(bg_flat, output_size=(h, w)).reshape(b, l, 1, h, w)
        fg_levels.append(fg_lvl)
        bg_levels.append(bg_lvl)
    return fg_levels, bg_levels
