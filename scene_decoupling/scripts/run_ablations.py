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
import subprocess
import time
from datetime import datetime

from scene_decoupling.src.common.config import load_config, resolve_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run key scene-decoupling ablations')
    p.add_argument('--config', nargs='+', required=True, help='Base config list (without ablation yaml)')
    p.add_argument(
        '--ablation-set',
        default='key_components',
        choices=['all', 'mask', 'mask_two', 'mask_align', 'decouple', 'compression', 'queue', 'key_components'],
    )
    p.add_argument('--override', nargs='*', default=[])
    p.add_argument('--max-runs', type=int, default=0, help='0 means run all in selected set')
    p.add_argument('--output', default='./scene_decoupling/outputs/cache/scene_decoupling_ablation_report.json')
    p.add_argument(
        '--resume',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Resume from existing report and skip successful completed runs.',
    )
    p.add_argument('--launcher', choices=['python', 'torchrun'], default='torchrun')
    p.add_argument('--nproc-per-node', type=int, default=2, help='Processes per node for torchrun.')
    p.add_argument('--master-port-base', type=int, default=29600, help='Base master port for sequential runs.')
    p.add_argument(
        '--cuda-visible-devices',
        default=None,
        help='GPU ids for training, e.g. 0,1. Default: inherit current environment.',
    )
    p.add_argument('--omp-num-threads', type=int, default=8)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def build_plan(ablation_set: str) -> list[dict[str, object]]:
    all_sets: dict[str, list[dict[str, object]]] = {
        'mask': [
            {'name': 'mask_no.yaml', 'config': 'mask_no.yaml', 'extra_overrides': []},
            {'name': 'mask_random.yaml', 'config': 'mask_random.yaml', 'extra_overrides': []},
            {'name': 'mask_skeleton.yaml', 'config': 'mask_skeleton.yaml', 'extra_overrides': []},
        ],
        # Run only non-skeleton mask variants; useful when skeleton-guided
        # baseline already exists and we only need NoMask/RandomMask.
        'mask_two': [
            {'name': 'mask_no.yaml', 'config': 'mask_no.yaml', 'extra_overrides': []},
            {'name': 'mask_random.yaml', 'config': 'mask_random.yaml', 'extra_overrides': []},
        ],
        # Mask alignment strategy ablation (max already in main method).
        'mask_align': [
            {'name': 'mask_align_avg.yaml', 'config': 'mask_align_avg.yaml', 'extra_overrides': []},
            {'name': 'mask_align_conv.yaml', 'config': 'mask_align_conv.yaml', 'extra_overrides': []},
        ],
        'decouple': [
            {'name': 'decouple_mixed.yaml', 'config': 'decouple_mixed.yaml', 'extra_overrides': []},
            {'name': 'decouple_action_only.yaml', 'config': 'decouple_action_only.yaml', 'extra_overrides': []},
            {'name': 'decouple_scene_only.yaml', 'config': 'decouple_scene_only.yaml', 'extra_overrides': []},
            {'name': 'decouple_dual.yaml', 'config': 'decouple_dual.yaml', 'extra_overrides': []},
        ],
        'compression': [
            {'name': 'compression_all_high.yaml', 'config': 'compression_all_high.yaml', 'extra_overrides': []},
            {'name': 'compression_all_low.yaml', 'config': 'compression_all_low.yaml', 'extra_overrides': []},
            {'name': 'compression_reverse_mixed.yaml', 'config': 'compression_reverse_mixed.yaml', 'extra_overrides': []},
            {'name': 'compression_default.yaml', 'config': 'compression_default.yaml', 'extra_overrides': []},
        ],
        'queue': [],
        # Requested key-component ablation set (exclude baseline/dual):
        # A: -scene (action_only), B: -action (scene_only), C: mixed(no-decouple)
        'key_components': [
            {'name': 'variantA_no_scene', 'config': 'decouple_action_only.yaml', 'extra_overrides': []},
            {'name': 'variantB_no_action', 'config': 'decouple_scene_only.yaml', 'extra_overrides': []},
            {'name': 'variantC_no_decouple', 'config': 'decouple_mixed.yaml', 'extra_overrides': []},
        ],
    }

    # Full queue grid sweep: action_len x scene_len.
    for action_len in [12, 24, 36]:
        for scene_len in [36, 72, 96]:
            all_sets['queue'].append(
                {
                    'name': f'queue_a{action_len}_s{scene_len}',
                    'config': 'queue_action24.yaml',  # reusable base yaml; overrides set exact pair
                    'extra_overrides': [
                        f'model.memory.action_len={action_len}',
                        f'model.memory.scene_len={scene_len}',
                        f'experiment.name=scene_decoupling_ablation_queue_a{action_len}_s{scene_len}',
                    ],
                }
            )

    if ablation_set == 'all':
        out: list[dict[str, object]] = []
        for k in ['mask', 'decouple', 'compression', 'queue']:
            out.extend(all_sets[k])
        return out
    return all_sets[ablation_set]


def _list_run_dirs(output_root: Path, exp_name: str) -> list[Path]:
    if not output_root.exists():
        return []
    prefix = f'{exp_name}_'
    return sorted([p for p in output_root.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def _resolve_cfg(root: Path, cfg_paths: list[str], overrides: list[str]) -> dict:
    cfg = resolve_paths(load_config(cfg_paths, overrides), project_root=root)
    return cfg


def _extract_summary_metrics(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / 'summary.json'
    if not summary_path.exists():
        return {}
    obj = json.loads(summary_path.read_text(encoding='utf-8'))
    best = obj.get('best_metrics', {})
    best_acc = best.get('best_acc', {})
    best_f1 = best.get('best_f1', {})
    last = obj.get('last_epoch_metrics', {})
    return {
        'monitor': obj.get('monitor'),
        'best_metric': obj.get('best_metric'),
        'best_threshold': obj.get('best_threshold'),
        'best_acc': best_acc.get('acc'),
        'best_f1': best_f1.get('f1'),
        'best_precision': best_acc.get('precision'),
        'best_recall': best_acc.get('recall'),
        'best_checkpoint': obj.get('best_checkpoint'),
        'last_epoch': last.get('epoch'),
        'last_val_loss': (((last.get('val') or {}).get('val_loss')) if isinstance(last, dict) else None),
        'last_val_acc_best': ((((last.get('val') or {}).get('best_acc') or {}).get('acc')) if isinstance(last, dict) else None),
        'last_val_f1_best': ((((last.get('val') or {}).get('best_f1') or {}).get('f1')) if isinstance(last, dict) else None),
    }


def _write_metrics_csv(path: Path, runs: list[dict[str, object]]) -> None:
    headers = [
        'name',
        'returncode',
        'run_dir',
        'monitor',
        'best_metric',
        'best_threshold',
        'best_acc',
        'best_f1',
        'best_precision',
        'best_recall',
        'last_epoch',
        'last_val_loss',
        'last_val_acc_best',
        'last_val_f1_best',
        'best_checkpoint',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in runs:
            m = dict(r.get('summary_metrics', {}))
            writer.writerow(
                {
                    'name': r.get('name', ''),
                    'returncode': r.get('returncode', ''),
                    'run_dir': r.get('run_dir', ''),
                    'monitor': m.get('monitor', ''),
                    'best_metric': m.get('best_metric', ''),
                    'best_threshold': m.get('best_threshold', ''),
                    'best_acc': m.get('best_acc', ''),
                    'best_f1': m.get('best_f1', ''),
                    'best_precision': m.get('best_precision', ''),
                    'best_recall': m.get('best_recall', ''),
                    'last_epoch': m.get('last_epoch', ''),
                    'last_val_loss': m.get('last_val_loss', ''),
                    'last_val_acc_best': m.get('last_val_acc_best', ''),
                    'last_val_f1_best': m.get('last_val_f1_best', ''),
                    'best_checkpoint': m.get('best_checkpoint', ''),
                }
            )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def _persist_report(output_json: Path, report: dict[str, object]) -> None:
    _write_json_atomic(output_json, report)
    _write_metrics_csv(output_json.with_suffix('.csv'), report.get('runs', []))


def _load_existing_report(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    runs = obj.get('runs', [])
    if not isinstance(runs, list):
        obj['runs'] = []
    return obj


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[2]
    ablation_dir = root / 'scene_decoupling' / 'configs' / 'ablation'
    train_script = root / 'scene_decoupling' / 'scripts' / 'train.py'
    out = Path(args.output)
    if not out.is_absolute():
        out = (root / out).resolve()

    plan = build_plan(args.ablation_set)
    if args.max_runs > 0:
        plan = plan[: args.max_runs]
    if int(args.nproc_per_node) <= 0:
        raise ValueError(f'--nproc-per-node must be > 0, got {args.nproc_per_node}')
    if int(args.master_port_base) <= 0:
        raise ValueError(f'--master-port-base must be > 0, got {args.master_port_base}')
    if args.launcher == 'torchrun' and shutil.which('torchrun') is None:
        raise RuntimeError('torchrun not found in PATH. Activate your venv with torch installed first.')

    report: dict[str, object]
    if args.resume:
        existing = _load_existing_report(out)
    else:
        existing = None

    if existing is not None:
        report = existing
        report['resumed_at'] = datetime.now().isoformat(timespec='seconds')
        report['ablation_set'] = args.ablation_set
        report['launcher'] = args.launcher
        report['nproc_per_node'] = int(args.nproc_per_node)
        report['master_port_base'] = int(args.master_port_base)
        report['cuda_visible_devices'] = (
            str(args.cuda_visible_devices)
            if args.cuda_visible_devices is not None
            else str(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
        )
        report['base_configs'] = args.config
        report['overrides'] = args.override
        runs = report.get('runs', [])
        if not isinstance(runs, list):
            runs = []
        report['runs'] = runs
    else:
        report = {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'ablation_set': args.ablation_set,
            'launcher': args.launcher,
            'nproc_per_node': int(args.nproc_per_node),
            'master_port_base': int(args.master_port_base),
            'cuda_visible_devices': (
                str(args.cuda_visible_devices)
                if args.cuda_visible_devices is not None
                else str(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
            ),
            'base_configs': args.config,
            'overrides': args.override,
            'runs': [],
        }

    runs = report['runs']
    done_names = {
        str(r.get('name'))
        for r in runs
        if isinstance(r, dict) and int(r.get('returncode', 1)) == 0 and str(r.get('name', '')).strip()
    }
    _persist_report(out, report)

    for item in plan:
        name = str(item['name'])
        if args.resume and name in done_names:
            print(f'[ablation] skip completed: {name}')
            continue

        cfg_path = ablation_dir / str(item['config'])
        extra_overrides = [str(x) for x in item.get('extra_overrides', [])]
        train_args = [
            str(train_script),
            '--config',
            *args.config,
            str(cfg_path),
        ]
        merged_overrides = [*args.override, *extra_overrides]
        if merged_overrides:
            train_args.extend(['--override', *merged_overrides])

        if args.launcher == 'torchrun':
            master_port = int(args.master_port_base) + len(runs)
            cmd = [
                'torchrun',
                '--standalone',
                '--nnodes=1',
                f'--nproc_per_node={int(args.nproc_per_node)}',
                f'--master_port={master_port}',
                *train_args,
            ]
        else:
            cmd = [sys.executable, *train_args]

        # Resolve output_root + exp_name so we can map command -> produced run dir.
        cfg_for_run = _resolve_cfg(
            root=root,
            cfg_paths=[*args.config, str(cfg_path)],
            overrides=merged_overrides,
        )
        output_root = Path(str(cfg_for_run['paths']['output_root'])).resolve()
        exp_name = str(cfg_for_run.get('experiment', {}).get('name', 'scene_decoupling'))
        before_dirs = {p.name for p in _list_run_dirs(output_root, exp_name)}

        print('[ablation] running:', ' '.join(cmd))
        entry: dict[str, object] = {
            'name': name,
            'config': str(cfg_path),
            'extra_overrides': extra_overrides,
            'experiment_name': exp_name,
            'output_root': str(output_root),
            'cmd': cmd,
            'returncode': None,
            'run_dir': '',
            'summary_path': '',
            'summary_metrics': {},
        }

        if args.dry_run:
            entry['returncode'] = 0
            runs.append(entry)
            _persist_report(out, report)
            continue

        t0 = time.time()
        env = dict(os.environ)
        if args.cuda_visible_devices is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(args.cuda_visible_devices)
        env['OMP_NUM_THREADS'] = str(int(args.omp_num_threads))
        proc = subprocess.run(cmd, cwd=str(root), check=False, env=env)
        entry['returncode'] = int(proc.returncode)
        after_dirs = _list_run_dirs(output_root, exp_name)
        created = [p for p in after_dirs if p.name not in before_dirs and p.stat().st_mtime >= (t0 - 5)]
        run_dir = created[-1] if created else (after_dirs[-1] if after_dirs else None)
        if run_dir is not None:
            entry['run_dir'] = str(run_dir.resolve())
            summary_path = run_dir / 'summary.json'
            entry['summary_path'] = str(summary_path.resolve()) if summary_path.exists() else ''
            entry['summary_metrics'] = _extract_summary_metrics(run_dir)
        runs.append(entry)
        _persist_report(out, report)

    failed = [r for r in runs if isinstance(r, dict) and r.get('returncode') != 0]
    csv_path = out.with_suffix('.csv')
    print(
        json.dumps(
            {
                'output': str(out),
                'metrics_csv': str(csv_path),
                'num_runs': len(runs),
                'num_failed': len(failed),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
