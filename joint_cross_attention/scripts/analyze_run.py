#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Analyze one joint-model run')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output', default='')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = _load_jsonl(run_dir / 'metrics.jsonl')
    best = max(rows, key=lambda x: ((x['val']['best_acc']['acc']), (x['val']['best_acc']['f1'])))
    report = {
        'run_dir': str(run_dir.resolve()),
        'epochs': len(rows),
        'best_epoch': int(best['epoch']),
        'best_val_acc': best['val']['best_acc'],
        'alpha_mean_best_epoch': best['val'].get('alpha_mean'),
        'beta_mean_best_epoch': best['val'].get('beta_mean'),
        'gamma_mean_best_epoch': best['val'].get('gamma_mean'),
        'fg_ratio_mean_best_epoch': best['val'].get('fg_ratio_mean'),
        'mask_overlap_mean_best_epoch': best['val'].get('mask_overlap_mean'),
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
