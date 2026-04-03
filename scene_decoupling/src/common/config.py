from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ConfigDict = dict[str, Any]


def _deep_merge(a: ConfigDict, b: ConfigDict) -> ConfigDict:
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _set_dotted(cfg: ConfigDict, key: str, value: Any) -> None:
    node = cfg
    parts = key.split('.')
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def load_yaml(path: str | Path) -> ConfigDict:
    with Path(path).open('r', encoding='utf-8') as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f'Config must be a mapping: {path}')
    return obj


def parse_overrides(overrides: list[str] | None) -> ConfigDict:
    out: ConfigDict = {}
    if not overrides:
        return out
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Invalid override: {item}')
        k, raw = item.split('=', 1)
        _set_dotted(out, k.strip(), yaml.safe_load(raw))
    return out


def load_config(paths: list[str | Path], overrides: list[str] | None = None) -> ConfigDict:
    if not paths:
        raise ValueError('No config paths provided')
    cfg: ConfigDict = {}
    for p in paths:
        cfg = _deep_merge(cfg, load_yaml(p))
    cfg = _deep_merge(cfg, parse_overrides(overrides))
    return cfg


def resolve_paths(cfg: ConfigDict, project_root: str | Path) -> ConfigDict:
    root = Path(project_root).resolve()
    out = deepcopy(cfg)
    paths = out.setdefault('paths', {})
    for k in ['output_root', 'cache_root', 'index_file']:
        if k in paths:
            p = Path(paths[k])
            if not p.is_absolute():
                paths[k] = str((root / p).resolve())
    for k in ['video_root', 'pose_root']:
        if k in paths:
            paths[k] = str(Path(paths[k]).resolve())
    return out
