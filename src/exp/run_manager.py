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
    metrics_jsonl: Path
    config_snapshot: Path
    summary_json: Path
    pitfall_log: Path


class RunManager:
    def __init__(self, output_root: str, exp_name: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{exp_name}_{ts}"
        self.run_dir = Path(output_root) / self.run_name
        self.paths = RunPaths(
            run_dir=self.run_dir,
            ckpt_dir=self.run_dir / "checkpoints",
            fig_dir=self.run_dir / "figures",
            metrics_jsonl=self.run_dir / "metrics.jsonl",
            config_snapshot=self.run_dir / "config_snapshot.yaml",
            summary_json=self.run_dir / "summary.json",
            pitfall_log=self.run_dir / "pitfalls.md",
        )
        self._init_dirs()

    def _init_dirs(self) -> None:
        self.paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.paths.fig_dir.mkdir(parents=True, exist_ok=True)

    def dump_config(self, cfg: dict[str, Any]) -> None:
        with self.paths.config_snapshot.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)

    def append_metrics(self, payload: dict[str, Any]) -> None:
        with self.paths.metrics_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def dump_summary(self, payload: dict[str, Any]) -> None:
        with self.paths.summary_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

    def append_pitfall(self, text: str) -> None:
        with self.paths.pitfall_log.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
