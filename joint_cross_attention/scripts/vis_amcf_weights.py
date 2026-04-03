#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import os
import shutil
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from matplotlib import font_manager
except Exception:  # pragma: no cover
    font_manager = None


MODALITIES: list[tuple[str, str]] = [
    ("alpha", "α"),
    ("beta", "β"),
    ("gamma", "γ"),
]
MODALITY_LABEL_MAP = {k: v for k, v in MODALITIES}
BASE_COLOR = (42, 92, 170)
BG_COLOR = (250, 250, 250)
BORDER_COLOR = (170, 176, 186)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-ready visualization for joint-model alpha/beta/gamma weights")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", default="", help="Joint-model run directory containing best_acc_eval.json or val_predictions_epoch_*.json")
    src.add_argument("--predictions-json", default="", help="Path to best_acc_eval.json or val_predictions_epoch_*.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-id", nargs="*", default=[], help="Optional explicit sample ids.")
    p.add_argument("--per-group", type=int, default=3, help="Auto mode: number of selected samples for each weight group.")
    p.add_argument("--thumb-width", type=int, default=720, help="Base frame width before render scaling.")
    p.add_argument("--thumb-height", type=int, default=405, help="Base frame height before render scaling.")
    p.add_argument("--frame-ratio", type=float, default=0.5, help="Representative frame position in [0,1].")
    p.add_argument("--render-scale", type=float, default=2.0, help="Render scale for high-resolution PNG output.")
    p.add_argument("--selection-mode", choices=["topk_per_weight", "dominant"], default="topk_per_weight")
    p.add_argument("--min-dominance-gap", type=float, default=0.0)
    p.add_argument("--correct-only", action=argparse.BooleanOptionalAction, default=True, help="Prefer only correctly predicted samples.")
    p.add_argument("--group-dominant-only", action=argparse.BooleanOptionalAction, default=True, help="When selecting top-k per weight, prefer samples whose dominant modality matches the group.")
    p.add_argument("--video-export", choices=["none", "symlink", "copy"], default="symlink", help="How to export the original video into each sample folder.")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions_json:
        path = Path(args.predictions_json)
        if not path.exists():
            raise FileNotFoundError(f"predictions json not found: {path}")
        return path

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    best = run_dir / "best_acc_eval.json"
    if best.exists():
        return best
    candidates = sorted(run_dir.glob("val_predictions_epoch_*.json"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No prediction file found under {run_dir}")


def _times_font(size: int, italic: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str] = []
    if font_manager is not None:
        wanted = ["Times New Roman", "Times New Roman Italic"] if italic else ["Times New Roman"]
        for name in wanted:
            try:
                candidates.append(font_manager.findfont(name, fallback_to_default=False))
            except Exception:
                pass
        fallback = ["Liberation Serif Italic", "Liberation Serif"] if italic else ["Liberation Serif"]
        for name in fallback:
            try:
                candidates.append(font_manager.findfont(name, fallback_to_default=False))
            except Exception:
                pass
    candidates.extend(
        [
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf" if italic else "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf" if italic else "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        ]
    )
    for path in candidates:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _prepare_predictions(predictions: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in predictions:
        item = dict(rec)
        item["pred_label"] = 1 if float(item.get("prob", 0.0)) >= float(threshold) else 0
        item["is_correct"] = int(item["pred_label"]) == int(item.get("label", 0))
        vals = {k: float(item.get(k, 0.0)) for k, _ in MODALITIES}
        order = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
        item["dominant"] = order[0][0]
        item["dominance_gap"] = float(order[0][1] - order[1][1]) if len(order) > 1 else 0.0
        out.append(item)
    return out


def _select_rows(predictions: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    source = [x for x in predictions if bool(x.get("is_correct", True))] if bool(args.correct_only) else list(predictions)
    if not source:
        source = list(predictions)

    if args.sample_id:
        by_id = {str(x["sample_id"]): x for x in predictions}
        chosen = [by_id[sid] for sid in args.sample_id if sid in by_id]
        if not chosen:
            raise ValueError("None of the provided --sample-id values were found.")
        return [{"row_key": "selected", "row_title": "selected", "records": chosen}]

    rows: list[dict[str, Any]] = []
    if args.selection_mode == "topk_per_weight":
        for key, label in MODALITIES:
            bucket_source = source
            if bool(args.group_dominant_only):
                matched = [x for x in source if x.get("dominant") == key]
                if matched:
                    bucket_source = matched
            bucket = sorted(
                bucket_source,
                key=lambda x: (float(x[key]), float(x["dominance_gap"]), float(x.get("prob", 0.0))),
                reverse=True,
            )
            chosen: list[dict[str, Any]] = []
            used: set[str] = set()
            for rec in bucket:
                sid = str(rec["sample_id"])
                if sid in used:
                    continue
                chosen.append(rec)
                used.add(sid)
                if len(chosen) >= max(1, int(args.per_group)):
                    break
            rows.append({"row_key": key, "row_title": f"top_{label}", "records": chosen})
        return rows

    for key, label in MODALITIES:
        bucket = [x for x in source if x["dominant"] == key and float(x["dominance_gap"]) >= float(args.min_dominance_gap)]
        bucket.sort(key=lambda x: (float(x[key]), float(x["dominance_gap"]), float(x.get("prob", 0.0))), reverse=True)
        rows.append({"row_key": key, "row_title": f"dominant_{label}", "records": bucket[: max(1, int(args.per_group))]})
    return rows


def _decode_frame(video_path: str, frame_ratio: float, target_hw: tuple[int, int]) -> Image.Image:
    w, h = target_hw
    canvas = Image.new("RGB", (w, h), BG_COLOR)
    path = Path(video_path)
    if not path.exists():
        return canvas

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return canvas

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_idx = 0 if total <= 1 else int(round((total - 1) * min(max(frame_ratio, 0.0), 1.0)))
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return canvas

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageOps.fit(img, (w, h), method=Image.Resampling.LANCZOS)


def _mix_blue(value: float) -> tuple[int, int, int]:
    v = float(min(max(value, 0.0), 1.0))
    bg = np.array([247, 249, 252], dtype=np.float32)
    fg = np.array(BASE_COLOR, dtype=np.float32)
    mixed = (1.0 - v) * bg + v * fg
    return tuple(int(round(x)) for x in mixed.tolist())


def _text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (255, 255, 255) if lum < 145 else (18, 18, 18)


def _draw_weight_boxes(
    draw: ImageDraw.ImageDraw,
    rec: dict[str, Any],
    panel_w: int,
    top_y: int,
    fonts: dict[str, ImageFont.ImageFont],
    scale: float,
) -> None:
    margin_x = int(round(24 * scale))
    gap = int(round(18 * scale))
    box_h = int(round(118 * scale))
    inner_w = panel_w - 2 * margin_x
    box_w = (inner_w - 2 * gap) // 3
    value_y = top_y + int(round(69 * scale))
    label_y = top_y + int(round(23 * scale))
    for idx, (key, label) in enumerate(MODALITIES):
        x0 = margin_x + idx * (box_w + gap)
        y0 = top_y
        x1 = x0 + box_w
        y1 = y0 + box_h
        fill = _mix_blue(float(rec.get(key, 0.0)))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=int(round(14 * scale)), fill=fill, outline=BORDER_COLOR, width=max(1, int(round(1.5 * scale))))
        tc = _text_color(fill)
        draw.text(((x0 + x1) / 2.0, label_y), label, fill=tc, font=fonts["box_label"], anchor="ma")
        draw.text(((x0 + x1) / 2.0, value_y), f"{float(rec.get(key, 0.0)):.3f}", fill=tc, font=fonts["box_value"], anchor="ma")


def _build_sample_panel(
    rec: dict[str, Any],
    thumb_hw: tuple[int, int],
    scale: float,
    show_compare_label: str = "",
) -> Image.Image:
    tw, th = thumb_hw
    title_h = int(round(52 * scale)) if show_compare_label else 0
    info_h = int(round(56 * scale))
    weights_h = int(round(148 * scale))
    panel = Image.new("RGB", (tw, th + title_h + info_h + weights_h), BG_COLOR)
    draw = ImageDraw.Draw(panel)
    fonts = {
        "compare": _times_font(int(round(26 * scale))),
        "info": _times_font(int(round(22 * scale))),
        "box_label": _times_font(int(round(24 * scale))),
        "box_value": _times_font(int(round(28 * scale))),
    }

    if show_compare_label:
        draw.text((tw / 2.0, int(round(28 * scale))), show_compare_label, fill=(25, 25, 25), font=fonts["compare"], anchor="ma")

    thumb = _decode_frame(str(rec.get("video_path", "")), frame_ratio=0.5, target_hw=(tw, th))
    y_img = title_h
    panel.paste(thumb, (0, y_img))
    draw.rectangle((0, y_img, tw - 1, y_img + th - 1), outline=BORDER_COLOR, width=max(1, int(round(1.4 * scale))))

    info_y = title_h + th + int(round(22 * scale))
    info_text = f"Label: y={int(rec.get('label', 0))}    Predictions: y={int(rec.get('pred_label', 0))}"
    draw.text((tw / 2.0, info_y), info_text, fill=(28, 28, 28), font=fonts["info"], anchor="ma")

    _draw_weight_boxes(draw=draw, rec=rec, panel_w=tw, top_y=title_h + th + info_h + int(round(8 * scale)), fonts=fonts, scale=scale)
    return panel


def _copy_or_link_video(video_path: str, dst_path: Path, mode: str) -> str:
    if mode == "none":
        return ""
    src = Path(video_path)
    if not src.exists():
        return ""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        if dst_path.is_dir():
            shutil.rmtree(dst_path)
        else:
            dst_path.unlink()
    if mode == "copy":
        shutil.copy2(src, dst_path)
    else:
        try:
            os.symlink(src.resolve(), dst_path)
        except OSError:
            shutil.copy2(src, dst_path)
    return str(dst_path.resolve())


def _resolve_group_layout(row: dict[str, Any], out_dir: Path) -> tuple[Path, str]:
    group_key = str(row.get("row_key", "selected"))
    if group_key in MODALITY_LABEL_MAP:
        return out_dir / f"{group_key}_high", f"High {MODALITY_LABEL_MAP[group_key]}"
    return out_dir / "samples", str(row.get("row_title", "Selected")).replace("_", " ").title()


def _save_grouped_samples(
    rows: list[dict[str, Any]],
    out_dir: Path,
    thumb_hw: tuple[int, int],
    scale: float,
    video_export: str,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    running_idx = 0
    for row in rows:
        group_key = str(row["row_key"])
        group_dir, panel_label = _resolve_group_layout(row=row, out_dir=out_dir)
        group_dir.mkdir(parents=True, exist_ok=True)
        for local_idx, rec in enumerate(row["records"], start=1):
            running_idx += 1
            sample_dir = group_dir / f"sample_{local_idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            panel = _build_sample_panel(rec, thumb_hw=thumb_hw, scale=scale, show_compare_label=panel_label)
            panel_path = sample_dir / "panel.png"
            panel.save(panel_path, compress_level=0)

            src_video = str(rec.get("video_path", ""))
            video_suffix = Path(src_video).suffix or ".mp4"
            exported_video = _copy_or_link_video(src_video, sample_dir / f"source_video{video_suffix}", mode=video_export)

            item = {
                "index": running_idx,
                "group": group_key,
                "group_rank": local_idx,
                "sample_dir": str(sample_dir.resolve()),
                "image_path": str(panel_path.resolve()),
                "exported_video_path": exported_video,
                "sample_id": rec.get("sample_id"),
                "video_id": rec.get("video_id"),
                "video_path": src_video,
                "label": rec.get("label"),
                "pred_label": rec.get("pred_label"),
                "prob": rec.get("prob"),
                "alpha": rec.get("alpha"),
                "beta": rec.get("beta"),
                "gamma": rec.get("gamma"),
                "dominant": rec.get("dominant"),
                "dominance_gap": rec.get("dominance_gap"),
            }
            (sample_dir / "meta.json").write_text(json.dumps(item, ensure_ascii=True, indent=2), encoding="utf-8")
            manifest.append(item)
    return manifest


def _pick_comparison_triplet(predictions: list[dict[str, Any]], correct_only: bool) -> list[tuple[str, dict[str, Any]]]:
    source = [x for x in predictions if bool(x.get("is_correct", True))] if correct_only else list(predictions)
    if not source:
        source = list(predictions)
    if not source:
        return []
    triplet: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()
    for key, symbol in MODALITIES:
        bucket = sorted(source, key=lambda x: (float(x[key]), float(x.get("dominance_gap", 0.0))), reverse=True)
        chosen = None
        for rec in bucket:
            sid = str(rec["sample_id"])
            if sid not in used:
                chosen = rec
                used.add(sid)
                break
        if chosen is None:
            chosen = bucket[0]
        triplet.append((f"Max {symbol}", chosen))
    return triplet


def _save_comparison_figure(
    predictions: list[dict[str, Any]],
    out_dir: Path,
    thumb_hw: tuple[int, int],
    scale: float,
    correct_only: bool,
) -> str:
    triplet = _pick_comparison_triplet(predictions, correct_only=correct_only)
    if len(triplet) < 2:
        return ""

    panels = [_build_sample_panel(rec, thumb_hw=thumb_hw, scale=scale, show_compare_label=label) for label, rec in triplet]
    gap = int(round(24 * scale))
    width = sum(x.size[0] for x in panels) + gap * (len(panels) - 1)
    height = max(x.size[1] for x in panels)
    canvas = Image.new("RGB", (width, height), BG_COLOR)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.size[0] + gap
    out_path = out_dir / "comparison_top_alpha_beta_gamma.png"
    canvas.save(out_path, compress_level=0)
    return str(out_path.resolve())


def _flatten_selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        for rec in row["records"]:
            sid = str(rec["sample_id"])
            if sid in seen:
                continue
            out.append(rec)
            seen.add(sid)
    return out


def _write_metadata(out_dir: Path, manifest: list[dict[str, Any]], rows: list[dict[str, Any]], comparison_path: str) -> None:
    summary = {
        "num_samples": int(len(manifest)),
        "rows": [{"row_key": row["row_key"], "row_title": row["row_title"], "count": int(len(row["records"]))} for row in rows],
        "comparison_path": comparison_path,
    }
    (out_dir / "amcf_weight_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    (out_dir / "amcf_weight_selection.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    with (out_dir / "amcf_weight_selection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "group",
                "group_rank",
                "sample_dir",
                "image_path",
                "exported_video_path",
                "sample_id",
                "video_id",
                "label",
                "pred_label",
                "prob",
                "alpha",
                "beta",
                "gamma",
                "dominant",
                "dominance_gap",
                "video_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)


def main() -> None:
    args = parse_args()
    pred_path = _resolve_predictions_path(args)
    payload = _load_json(pred_path)
    predictions = payload.get("predictions", payload)
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"No predictions found in {pred_path}")
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    best_acc = summary.get("best_acc", {}) if isinstance(summary, dict) else {}
    threshold = float(best_acc.get("threshold", 0.5))

    prepared = _prepare_predictions(predictions, threshold=threshold)
    rows = _select_rows(prepared, args)
    selected = _flatten_selected(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = float(max(args.render_scale, 1.0))
    thumb_hw = (int(round(int(args.thumb_width) * scale)), int(round(int(args.thumb_height) * scale)))
    manifest = _save_grouped_samples(rows=rows, out_dir=out_dir, thumb_hw=thumb_hw, scale=scale, video_export=str(args.video_export))
    comparison_path = _save_comparison_figure(predictions=prepared, out_dir=out_dir, thumb_hw=thumb_hw, scale=scale, correct_only=bool(args.correct_only))
    _write_metadata(out_dir=out_dir, manifest=manifest, rows=rows, comparison_path=comparison_path)

    print(
        json.dumps(
            {
                "predictions_json": str(pred_path.resolve()),
                "samples_root": str(out_dir.resolve()),
                "comparison_png": comparison_path,
                "selection_json": str((out_dir / "amcf_weight_selection.json").resolve()),
                "selection_csv": str((out_dir / "amcf_weight_selection.csv").resolve()),
                "summary_json": str((out_dir / "amcf_weight_summary.json").resolve()),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
