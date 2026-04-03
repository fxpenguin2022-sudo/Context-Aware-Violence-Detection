from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


def _deep_merge(base: ConfigDict, new: ConfigDict) -> ConfigDict:
    out = deepcopy(base)
    for k, v in new.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _set_dotted(cfg: ConfigDict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def load_yaml(path: str | Path) -> ConfigDict:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return obj


def parse_overrides(overrides: list[str] | None) -> ConfigDict:
    cfg: ConfigDict = {}
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        _set_dotted(cfg, key.strip(), value)
    return cfg


def load_config(config_paths: list[str | Path], overrides: list[str] | None = None) -> ConfigDict:
    if not config_paths:
        raise ValueError("At least one config file is required")
    merged: ConfigDict = {}
    for cfg_path in config_paths:
        merged = _deep_merge(merged, load_yaml(cfg_path))
    merged = _deep_merge(merged, parse_overrides(overrides))
    return merged


def resolve_paths(cfg: ConfigDict, project_root: str | Path) -> ConfigDict:
    root = Path(project_root).resolve()
    out = deepcopy(cfg)
    paths = out.setdefault("paths", {})

    for key in ["output_root", "cache_root", "index_file"]:
        if key in paths:
            p = Path(paths[key])
            if not p.is_absolute():
                paths[key] = str((root / p).resolve())

    for key in ["pose_root", "video_root"]:
        if key in paths:
            paths[key] = str(Path(paths[key]).resolve())

    return out
