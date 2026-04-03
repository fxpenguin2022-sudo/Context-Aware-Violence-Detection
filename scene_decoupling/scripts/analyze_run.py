#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description='Analyze one scene-decoupling run')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output', default='')
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    rows = read_jsonl(run_dir / 'metrics.jsonl')
    summary_file = run_dir / 'summary.json'
    monitor = 'f1'
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        monitor = str(summary.get('monitor', monitor))

    monitor_key = monitor.strip().lower()
    if monitor_key == 'val_loss':
        best = min(rows, key=lambda x: x['val'].get('val_loss', 1e9))
        best_metrics = best['val'].get('best_f1', best['val']['best'])
        last_metrics = rows[-1]['val'].get('best_f1', rows[-1]['val']['best'])
    elif monitor_key in {'acc', 'accuracy', 'val_acc'}:
        best = max(rows, key=lambda x: x['val'].get('best_acc', x['val']['best'])['acc'])
        best_metrics = best['val'].get('best_acc', best['val']['best'])
        last_metrics = rows[-1]['val'].get('best_acc', rows[-1]['val']['best'])
    elif monitor_key in {'balanced_acc', 'bacc', 'val_balanced_acc'}:
        best = max(rows, key=lambda x: x['val'].get('best_balanced_acc', x['val']['best'])['balanced_acc'])
        best_metrics = best['val'].get('best_balanced_acc', best['val']['best'])
        last_metrics = rows[-1]['val'].get('best_balanced_acc', rows[-1]['val']['best'])
    else:
        best = max(rows, key=lambda x: x['val'].get('best_f1', x['val']['best'])['f1'])
        best_metrics = best['val'].get('best_f1', best['val']['best'])
        last_metrics = rows[-1]['val'].get('best_f1', rows[-1]['val']['best'])
    report = {
        'run_dir': str(run_dir.resolve()),
        'epochs': len(rows),
        'best_epoch': int(best['epoch']),
        'monitor': monitor,
        'best_metrics': best_metrics,
        'last_metrics': last_metrics,
        'best_metrics_f1': best['val'].get('best_f1', best['val']['best']),
        'best_metrics_acc': best['val'].get('best_acc', best['val']['best']),
        'fg_ratio_mean_best_epoch': best['val'].get('fg_ratio_mean', None),
    }

    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
