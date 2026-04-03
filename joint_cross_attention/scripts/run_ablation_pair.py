#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime

from scene_decoupling.src.common.config import load_config, resolve_paths


ABLATONS = [
    ('no_sg_aca_concat', 'no_sg_aca_concat.yaml'),
    ('no_amcf_fixedavg', 'no_amcf_fixedavg.yaml'),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run the joint-model ablation pair sequentially')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--override', nargs='*', default=[])
    p.add_argument('--output', default='joint_cross_attention/outputs/cache/joint_model_ablation_pair_report.json')
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def _load_existing(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _list_run_dirs(output_root: Path, exp_name: str) -> list[Path]:
    if not output_root.exists():
        return []
    prefix = f'{exp_name}_'
    return sorted([p for p in output_root.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def _resolve_cfg(root: Path, cfg_paths: list[str], overrides: list[str]) -> dict:
    return resolve_paths(load_config(cfg_paths, overrides), project_root=root)


def _extract_summary(run_dir: Path) -> dict:
    path = run_dir / 'summary.json'
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding='utf-8'))
    best = obj.get('best_metrics', {})
    best_acc = best.get('best_acc', {})
    best_f1 = best.get('best_f1', {})
    return {
        'acc': best_acc.get('acc'),
        'f1': best_f1.get('f1'),
        'precision': best_acc.get('precision'),
        'recall': best_acc.get('recall'),
        'threshold': best_acc.get('threshold'),
        'best_checkpoint': obj.get('best_checkpoint'),
    }


def _write_csv(path: Path, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'returncode', 'run_dir', 'acc', 'f1', 'precision', 'recall', 'threshold', 'best_checkpoint'])
        writer.writeheader()
        for row in runs:
            s = row.get('summary_metrics', {})
            writer.writerow(
                {
                    'name': row.get('name', ''),
                    'returncode': row.get('returncode', ''),
                    'run_dir': row.get('run_dir', ''),
                    'acc': s.get('acc', ''),
                    'f1': s.get('f1', ''),
                    'precision': s.get('precision', ''),
                    'recall': s.get('recall', ''),
                    'threshold': s.get('threshold', ''),
                    'best_checkpoint': s.get('best_checkpoint', ''),
                }
            )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    train_script = root / 'joint_cross_attention' / 'scripts' / 'train.py'
    ablation_dir = root / 'joint_cross_attention' / 'configs' / 'ablation'
    out = Path(args.output)
    if not out.is_absolute():
        out = (root / out).resolve()

    report = _load_existing(out) if args.resume else None
    if report is None:
        report = {'created_at': datetime.now().isoformat(timespec='seconds'), 'runs': []}
    runs = report.get('runs', []) if isinstance(report.get('runs'), list) else []
    done = {
        str(r.get('name'))
        for r in runs
        if isinstance(r, dict)
        and int(r.get('returncode', 1)) == 0
        and str(r.get('run_dir', '')).strip()
    }

    for name, cfg_name in ABLATONS:
        if args.resume and name in done:
            print(f'[ablation] skip completed: {name}')
            continue
        cfg_path = ablation_dir / cfg_name
        merged_overrides = [*args.override]
        cfg_for_run = _resolve_cfg(root, [*args.config, str(cfg_path)], merged_overrides)
        output_root = Path(str(cfg_for_run['paths']['output_root'])).resolve()
        exp_name = str(cfg_for_run.get('experiment', {}).get('name', 'joint_model'))
        before_dirs = {p.name for p in _list_run_dirs(output_root, exp_name)}

        cmd = [sys.executable, str(train_script), '--config', *args.config, str(cfg_path)]
        if merged_overrides:
            cmd.extend(['--override', *merged_overrides])

        entry = {
            'name': name,
            'config': str(cfg_path),
            'returncode': None,
            'run_dir': '',
            'summary_metrics': {},
            'cmd': cmd,
        }
        if args.dry_run:
            entry['returncode'] = -99
            runs.append(entry)
            continue

        env = dict(os.environ)
        proc = subprocess.run(cmd, cwd=str(root), check=False, env=env)
        entry['returncode'] = int(proc.returncode)
        after_dirs = _list_run_dirs(output_root, exp_name)
        created = [p for p in after_dirs if p.name not in before_dirs]
        run_dir = created[-1] if created else (after_dirs[-1] if after_dirs else None)
        if run_dir is not None:
            entry['run_dir'] = str(run_dir.resolve())
            entry['summary_metrics'] = _extract_summary(run_dir)
        runs.append(entry)
        report['runs'] = runs
        _write_json(out, report)

    _write_csv(out.with_suffix('.csv'), runs)
    print(json.dumps({'output': str(out), 'csv': str(out.with_suffix('.csv'))}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
