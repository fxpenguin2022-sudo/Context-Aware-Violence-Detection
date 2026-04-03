from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunPaths:
    run_dir: Path
    ckpt_dir: Path
    fig_dir: Path
    tb_dir: Path
    config_snapshot: Path
    metrics_jsonl: Path
    summary_json: Path


class RunManager:
    def __init__(self, output_root: str, exp_name: str) -> None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f'{exp_name}_{ts}'
        run_dir = Path(output_root) / self.run_name
        self.paths = RunPaths(
            run_dir=run_dir,
            ckpt_dir=run_dir / 'checkpoints',
            fig_dir=run_dir / 'figures',
            tb_dir=run_dir / 'tensorboard',
            config_snapshot=run_dir / 'config_snapshot.yaml',
            metrics_jsonl=run_dir / 'metrics.jsonl',
            summary_json=run_dir / 'summary.json',
        )
        self.paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.paths.fig_dir.mkdir(parents=True, exist_ok=True)
        self.paths.tb_dir.mkdir(parents=True, exist_ok=True)

    def dump_config(self, cfg: dict[str, Any]) -> None:
        with self.paths.config_snapshot.open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)

    def append_metrics(self, row: dict[str, Any]) -> None:
        with self.paths.metrics_jsonl.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=True) + '\n')

    def dump_summary(self, obj: dict[str, Any]) -> None:
        with self.paths.summary_json.open('w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=True, indent=2)
