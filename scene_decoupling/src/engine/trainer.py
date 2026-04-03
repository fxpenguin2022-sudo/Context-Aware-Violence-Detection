from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scene_decoupling.src.common.checkpoint import save_checkpoint
from scene_decoupling.src.common.distributed import all_reduce_sum, barrier, is_main_process
from scene_decoupling.src.engine.evaluator import evaluate
from scene_decoupling.src.exp.artifact_writer import CsvWriter
from scene_decoupling.src.exp.run_manager import RunManager

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


@dataclass
class TrainArtifacts:
    best_metric: float
    best_threshold: float
    best_ckpt: str


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _split_param_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    back_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if '.backbone.' in name or name.startswith('backbone.') or name.startswith('module.backbone.'):
            back_params.append(param)
        else:
            head_params.append(param)

    if not head_params:
        head_params = back_params
        back_params = []

    return back_params, head_params


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    c = cfg['train']['optimizer']
    if c['name'].lower() != 'adamw':
        raise ValueError(f"Unsupported optimizer {c['name']}")

    lr_backbone = float(c.get('lr_backbone', c.get('lr', 1e-4)))
    lr_head = float(c.get('lr_head', c.get('lr', 1e-4)))
    weight_decay = float(c['weight_decay'])
    betas = tuple(c.get('betas', [0.9, 0.999]))

    back_params, head_params = _split_param_groups(model)
    groups = []
    if back_params:
        groups.append({'params': back_params, 'lr': lr_backbone, 'name': 'backbone'})
    if head_params:
        groups.append({'params': head_params, 'lr': lr_head, 'name': 'head'})

    return torch.optim.AdamW(groups, weight_decay=weight_decay, betas=betas)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any]):
    c = cfg['train']['scheduler']
    epochs = int(cfg['train']['epochs'])
    min_lr = float(c.get('min_lr', 1e-6))
    warmup = int(c.get('warmup_epochs', 0))

    base_lrs = [max(pg['lr'], 1e-12) for pg in optimizer.param_groups]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())
        # A shared multiplier, each param group keeps its own base lr.
        return cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # Clamp min lr manually in training loop.
    scheduler.min_lr = min_lr  # type: ignore[attr-defined]
    scheduler.base_lrs = base_lrs  # type: ignore[attr-defined]
    return scheduler


def _apply_scheduler_min_lr(optimizer: torch.optim.Optimizer, scheduler) -> None:
    min_lr = float(getattr(scheduler, 'min_lr', 0.0))
    for pg in optimizer.param_groups:
        pg['lr'] = max(float(pg['lr']), min_lr)


def _collect_final_hparams(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        'data': {
            'frame_size': int(cfg['data']['frame_size'][0]),
            'clip_len': int(cfg['data']['clip_len']),
            'clip_step': int(cfg['data']['clip_step']),
            'full_sequence': bool(cfg['data'].get('full_sequence', True)),
            'max_clips_train': int(cfg['data'].get('max_clips_train', 0)),
            'max_clips_eval': int(cfg['data'].get('max_clips_eval', 0)),
            'max_persons': int(cfg['data']['max_persons']),
            'num_keypoints': int(cfg['data']['num_keypoints']),
            'mask_sigma': float(cfg['data']['mask_sigma']),
            'mask_threshold': float(cfg['data']['mask_threshold']),
        },
        'model': {
            'backbone': str(cfg['model']['backbone']),
            'memvit_repo_path': str(cfg['model'].get('memvit_repo_path', '')),
            'backbone_pretrained_path': str(cfg['model'].get('backbone_pretrained_path', '')),
            'memvit_input_temporal': int(cfg['model'].get('memvit_input_temporal', 16)),
            'memvit_capture_blocks': list(cfg['model'].get('memvit_capture_blocks', [1, 3, 14])),
            'backbone_freeze_stages': int(cfg['model']['backbone_freeze_stages']),
            'proj_dim': int(cfg['model']['proj_dim']),
            'num_heads': int(cfg['model']['num_heads']),
            'attn_layers_per_stream': int(cfg['model']['attn_layers_per_stream']),
            'mlp_ratio': float(cfg['model']['mlp_ratio']),
            'attn_dropout': float(cfg['model']['attn_dropout']),
            'proj_dropout': float(cfg['model']['proj_dropout']),
            'fusion_hidden_dim': int(cfg['model']['fusion']['hidden_dim']),
            'fusion_dropout': float(cfg['model']['fusion']['dropout']),
            'decouple_mode': str(cfg['model']['decouple']['mode']),
            'mask_mode': str(cfg['model']['mask']['mode']),
            'mask_align_strategy': str(cfg['model']['mask'].get('align_strategy', 'max')),
            'mask_enforce_complement': bool(cfg['model']['mask'].get('enforce_complement', False)),
        },
        'memory': {
            'action_len': int(cfg['model']['memory']['action_len']),
            'scene_len': int(cfg['model']['memory']['scene_len']),
            'action_pool': list(cfg['model']['memory']['action_pool']),
            'scene_pool': list(cfg['model']['memory']['scene_pool']),
            'stop_grad_memory': bool(cfg['model']['memory']['stop_grad_memory']),
            'compression_mode': str(cfg['model']['memory'].get('compression_mode', 'default')),
        },
        'loss': {
            'name': str(cfg['loss']['name']),
            'pos_weight': float(cfg['loss'].get('pos_weight', 1.0)),
            'focal_gamma': float(cfg['loss'].get('focal_gamma', 0.0)),
            'label_smoothing': float(cfg['loss'].get('label_smoothing', 0.0)),
            'sep_weight': float(cfg['loss'].get('sep_weight', 0.0)),
            'overlap_weight': float(cfg['loss'].get('overlap_weight', 0.0)),
            'fg_ratio_weight': float(cfg['loss'].get('fg_ratio_weight', 0.0)),
            'fg_ratio_min': float(cfg['loss'].get('fg_ratio_min', 0.0)),
            'fg_ratio_max': float(cfg['loss'].get('fg_ratio_max', 1.0)),
            'constraint_warmup_epochs': int(cfg['loss'].get('constraint_warmup_epochs', 0)),
        },
        'train': {
            'epochs': int(cfg['train']['epochs']),
            'monitor': str(cfg['train']['early_stop']['monitor']),
            'patience': int(cfg['train']['early_stop']['patience']),
            'optimizer': str(cfg['train']['optimizer']['name']),
            'lr_backbone': float(cfg['train']['optimizer']['lr_backbone']),
            'lr_head': float(cfg['train']['optimizer']['lr_head']),
            'weight_decay': float(cfg['train']['optimizer']['weight_decay']),
            'betas': list(cfg['train']['optimizer']['betas']),
            'scheduler': str(cfg['train']['scheduler']['name']),
            'warmup_epochs': int(cfg['train']['scheduler']['warmup_epochs']),
            'min_lr': float(cfg['train']['scheduler']['min_lr']),
            'batch_size_per_gpu': int(cfg['train']['batch_size']),
            'eval_batch_size_per_gpu': int(cfg['train']['eval_batch_size']),
            'grad_accum_steps': int(cfg['train']['grad_accum_steps']),
            'grad_clip_norm': float(cfg['train']['grad_clip_norm']),
            'use_amp': bool(cfg['runtime'].get('use_amp', True)),
            'amp_dtype': str(cfg['runtime'].get('amp_dtype', 'bf16')),
        },
        'eval': {
            'fixed_threshold': float(cfg['eval']['threshold']['fixed']),
            'scan_min': float(cfg['eval']['threshold']['scan_min']),
            'scan_max': float(cfg['eval']['threshold']['scan_max']),
            'scan_steps': int(cfg['eval']['threshold']['scan_steps']),
        },
    }


def _write_final_hparams(run_manager: RunManager, cfg: dict[str, Any]) -> None:
    payload = _collect_final_hparams(cfg)
    json_path = run_manager.paths.run_dir / 'final_hparams.json'
    md_path = run_manager.paths.run_dir / 'final_hparams.md'

    with json_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    lines = ['# Final Hyperparameters', '']
    for section, values in payload.items():
        lines.append(f'## {section}')
        for k, v in values.items():
            lines.append(f'- `{k}`: `{v}`')
        lines.append('')
    md_path.write_text('\n'.join(lines), encoding='utf-8')


def _normalize_monitor(monitor: str) -> str:
    key = str(monitor).strip().lower()
    aliases = {
        'loss': 'val_loss',
        'val_loss': 'val_loss',
        'f1': 'f1',
        'val_f1': 'f1',
        'acc': 'acc',
        'accuracy': 'acc',
        'val_acc': 'acc',
        'balanced_acc': 'balanced_acc',
        'bacc': 'balanced_acc',
        'val_balanced_acc': 'balanced_acc',
    }
    if key not in aliases:
        raise ValueError(f'Unsupported early-stop monitor: {monitor}')
    return aliases[key]


def _select_monitor_record(monitor: str, ev_summary: dict[str, Any]) -> tuple[float, dict[str, float]]:
    best_f1 = ev_summary.get('best_f1', ev_summary['best'])
    best_acc = ev_summary.get('best_acc', best_f1)
    best_balanced_acc = ev_summary.get('best_balanced_acc', best_f1)

    if monitor == 'val_loss':
        return float(ev_summary['val_loss']), best_f1
    if monitor == 'f1':
        return float(best_f1['f1']), best_f1
    if monitor == 'acc':
        return float(best_acc['acc']), best_acc
    if monitor == 'balanced_acc':
        return float(best_balanced_acc['balanced_acc']), best_balanced_acc
    raise ValueError(f'Unsupported monitor: {monitor}')


def _monitor_checkpoint_name(monitor: str) -> str:
    return 'best_val_loss.pt' if monitor == 'val_loss' else f'best_{monitor}.pt'


def _distributed_max_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _constraint_warmup_scale(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, float(epoch - 1) / float(max(1, warmup_epochs)))))


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
    logger,
    constraint_scale: float = 1.0,
) -> dict[str, float]:
    model.train()
    accum = int(cfg['train'].get('grad_accum_steps', 1))
    grad_clip = float(cfg['train'].get('grad_clip_norm', 0.0))
    use_amp = bool(cfg['runtime'].get('use_amp', True))
    amp_dtype = torch.bfloat16 if cfg['runtime'].get('amp_dtype', 'bf16') == 'bf16' else torch.float16
    max_steps = int(cfg['train'].get('max_steps_per_epoch', 0))

    show_pbar = bool(cfg['logging'].get('progress_bar', True)) and is_main_process() and tqdm is not None
    show_step_logs_with_pbar = bool(cfg['logging'].get('step_log_with_pbar', False))
    show_pbar_postfix = bool(cfg['logging'].get('pbar_postfix', True))
    log_interval = max(1, int(cfg['logging'].get('log_interval', 20)))
    iterator = loader
    if show_pbar:
        total = min(len(loader), max_steps) if max_steps > 0 else len(loader)
        iterator = tqdm(
            loader,
            total=total,
            desc=f'Train2 E{epoch:03d}',
            leave=False,
            dynamic_ncols=True,
            miniters=1,
            file=sys.stdout,
        )

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_loss_video = 0.0
    running_loss_sep = 0.0
    running_loss_overlap = 0.0
    running_loss_fg_ratio = 0.0
    running_fg = 0.0
    running_overlap = 0.0
    steps = 0

    for step, batch in enumerate(iterator):
        clips = batch['clips'].to(device, non_blocking=True)
        poses = batch['poses'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        clip_valid_mask = batch['clip_valid_mask'].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
            out = model(clips, poses, clip_valid_mask=clip_valid_mask)
            try:
                loss_dict = criterion(
                    out['video_logit'],
                    labels,
                    f_evol=out.get('f_evol'),
                    f_scene=out.get('f_scene'),
                    mask_overlap=out.get('mask_overlap'),
                    fg_ratio=out.get('fg_ratio'),
                    weight_scale=constraint_scale,
                )
            except TypeError:
                loss_dict = criterion(out['video_logit'], labels)
            loss = loss_dict['loss'] / accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss_dict['loss'].detach().item())
        loss_video_t = loss_dict.get('loss_video', loss_dict['loss'])
        loss_sep_t = loss_dict.get('loss_sep', loss_dict['loss'].new_zeros(()))
        loss_overlap_t = loss_dict.get('loss_overlap', loss_dict['loss'].new_zeros(()))
        loss_fg_ratio_t = loss_dict.get('loss_fg_ratio', loss_dict['loss'].new_zeros(()))
        running_loss_video += float(loss_video_t.detach().item())
        running_loss_sep += float(loss_sep_t.detach().item())
        running_loss_overlap += float(loss_overlap_t.detach().item())
        running_loss_fg_ratio += float(loss_fg_ratio_t.detach().item())
        running_fg += float(out['fg_ratio'].detach().mean().item())
        running_overlap += float(out['mask_overlap'].detach().mean().item())
        steps += 1

        avg_loss = running_loss / max(1, steps)
        avg_fg = running_fg / max(1, steps)
        avg_sep = running_loss_sep / max(1, steps)
        avg_overlap = running_overlap / max(1, steps)

        if show_pbar and show_pbar_postfix:
            iterator.set_postfix(
                {'loss': f'{avg_loss:.4f}', 'fg': f'{avg_fg:.4f}', 'sep': f'{avg_sep:.4f}', 'ov': f'{avg_overlap:.4f}'},
                refresh=True,
            )

        if is_main_process() and step % log_interval == 0 and (not show_pbar or show_step_logs_with_pbar):
            logger.info(
                'Epoch %d Step %d/%d | loss=%.4f fg_ratio=%.4f mask_overlap=%.4f scale=%.2f',
                epoch,
                step,
                len(loader),
                avg_loss,
                avg_fg,
                avg_overlap,
                constraint_scale,
            )

        if max_steps > 0 and (step + 1) >= max_steps:
            break

    if steps % accum != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    stat = torch.tensor(
        [
            running_loss,
            running_loss_video,
            running_loss_sep,
            running_loss_overlap,
            running_loss_fg_ratio,
            running_fg,
            running_overlap,
            steps,
        ],
        dtype=torch.float64,
        device=device,
    )
    stat = all_reduce_sum(stat)
    denom = max(float(stat[7].item()), 1.0)
    return {
        'loss': float(stat[0].item() / denom),
        'loss_video': float(stat[1].item() / denom),
        'loss_sep': float(stat[2].item() / denom),
        'loss_overlap': float(stat[3].item() / denom),
        'loss_fg_ratio': float(stat[4].item() / denom),
        'fg_ratio': float(stat[5].item() / denom),
        'mask_overlap': float(stat[6].item() / denom),
        'constraint_scale': float(constraint_scale),
    }


def fit(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    run_manager: RunManager,
    cfg: dict[str, Any],
    device: torch.device,
    logger,
) -> TrainArtifacts:
    use_amp = bool(cfg['runtime'].get('use_amp', True))
    amp_dtype = torch.bfloat16 if cfg['runtime'].get('amp_dtype', 'bf16') == 'bf16' else torch.float16
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    else:
        scaler = None

    history = CsvWriter(
        path=str(run_manager.paths.run_dir / 'history.csv'),
        headers=[
            'epoch',
            'train_loss',
            'train_loss_video',
            'train_loss_sep',
            'train_loss_overlap',
            'train_loss_fg_ratio',
            'train_constraint_scale',
            'train_fg_ratio',
            'train_mask_overlap',
            'val_loss',
            'val_acc_fixed',
            'val_precision_fixed',
            'val_recall_fixed',
            'val_f1_fixed',
            'val_acc_best',
            'val_precision_best',
            'val_recall_best',
            'val_f1_best',
            'best_threshold',
            'fg_ratio_mean',
            'val_mask_overlap_mean',
            'lr_backbone',
            'lr_head',
            'train_time_s',
            'eval_time_s',
            'epoch_time_s',
            'max_gpu_mem_gb',
        ],
    )

    tb = None
    if is_main_process() and bool(cfg['logging'].get('tensorboard', True)) and SummaryWriter is not None:
        tb = SummaryWriter(log_dir=str(run_manager.paths.tb_dir))

    monitor = _normalize_monitor(str(cfg['train']['early_stop'].get('monitor', 'val_loss')))
    best_metric = float('inf') if monitor == 'val_loss' else -1.0
    best_threshold = float(cfg['eval']['threshold']['fixed'])
    best_ckpt = ''
    best_ckpt_name = _monitor_checkpoint_name(monitor)
    best_metrics: dict[str, Any] = {}
    last_epoch_metrics: dict[str, Any] = {}
    no_improve = 0
    epochs = int(cfg['train']['epochs'])
    patience = int(cfg['train']['early_stop'].get('patience', 12))
    constraint_warmup_epochs = int(cfg['loss'].get('constraint_warmup_epochs', 0))

    if is_main_process():
        _write_final_hparams(run_manager, cfg)

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        epoch_t0 = time.perf_counter()
        constraint_scale = _constraint_warmup_scale(epoch=epoch, warmup_epochs=constraint_warmup_epochs)

        train_t0 = time.perf_counter()
        tr = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            epoch,
            logger,
            constraint_scale=constraint_scale,
        )
        train_time_s = time.perf_counter() - train_t0

        eval_t0 = time.perf_counter()
        ev = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            fixed_threshold=float(cfg['eval']['threshold']['fixed']),
            scan_cfg=cfg['eval']['threshold'],
            criterion=criterion,
            loss_weight_scale=constraint_scale,
            max_batches=int(cfg['eval'].get('max_batches', 0)),
            show_progress=bool(cfg['logging'].get('progress_bar', True)),
        )
        eval_time_s = time.perf_counter() - eval_t0
        epoch_time_s = time.perf_counter() - epoch_t0
        max_gpu_mem_local = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0**3)) if device.type == 'cuda' else 0.0
        )
        max_gpu_mem_gb = _distributed_max_scalar(max_gpu_mem_local, device=device)

        fixed = ev.summary['fixed']
        best_f1 = ev.summary.get('best_f1', ev.summary['best'])
        best_acc = ev.summary.get('best_acc', best_f1)
        monitor_metric, monitor_record = _select_monitor_record(monitor, ev.summary)
        last_epoch_metrics = {
            'epoch': int(epoch),
            'train': {
                'loss': float(tr['loss']),
                'loss_video': float(tr['loss_video']),
                'loss_sep': float(tr['loss_sep']),
                'loss_overlap': float(tr['loss_overlap']),
                'loss_fg_ratio': float(tr['loss_fg_ratio']),
                'constraint_scale': float(tr['constraint_scale']),
                'fg_ratio': float(tr['fg_ratio']),
                'mask_overlap': float(tr['mask_overlap']),
            },
            'val': {
                'num_videos': int(ev.summary['num_videos']),
                'val_loss': float(ev.summary['val_loss']),
                'fixed': fixed,
                'best_f1': best_f1,
                'best_acc': best_acc,
                'best_balanced_acc': ev.summary.get('best_balanced_acc', best_acc),
                'fg_ratio_mean': float(ev.summary['fg_ratio_mean']),
                'mask_overlap_mean': float(ev.summary.get('mask_overlap_mean', 0.0)),
            },
            'perf': {
                'train_time_s': float(train_time_s),
                'eval_time_s': float(eval_time_s),
                'epoch_time_s': float(epoch_time_s),
                'max_gpu_mem_gb': float(max_gpu_mem_gb),
            },
        }

        lr_backbone = float(optimizer.param_groups[0]['lr'])
        lr_head = float(optimizer.param_groups[-1]['lr'])

        if is_main_process():
            history.write(
                {
                    'epoch': epoch,
                    'train_loss': tr['loss'],
                    'train_loss_video': tr['loss_video'],
                    'train_loss_sep': tr['loss_sep'],
                    'train_loss_overlap': tr['loss_overlap'],
                    'train_loss_fg_ratio': tr['loss_fg_ratio'],
                    'train_constraint_scale': tr['constraint_scale'],
                    'train_fg_ratio': tr['fg_ratio'],
                    'train_mask_overlap': tr['mask_overlap'],
                    'val_loss': ev.summary['val_loss'],
                    'val_acc_fixed': fixed['acc'],
                    'val_precision_fixed': fixed['precision'],
                    'val_recall_fixed': fixed['recall'],
                    'val_f1_fixed': fixed['f1'],
                    'val_acc_best': best_acc['acc'],
                    'val_precision_best': best_acc['precision'],
                    'val_recall_best': best_acc['recall'],
                    'val_f1_best': best_acc['f1'],
                    'best_threshold': best_acc['threshold'],
                    'fg_ratio_mean': ev.summary['fg_ratio_mean'],
                    'val_mask_overlap_mean': ev.summary.get('mask_overlap_mean', 0.0),
                    'lr_backbone': lr_backbone,
                    'lr_head': lr_head,
                    'train_time_s': train_time_s,
                    'eval_time_s': eval_time_s,
                    'epoch_time_s': epoch_time_s,
                    'max_gpu_mem_gb': max_gpu_mem_gb,
                }
            )

            run_manager.append_metrics(
                {
                    'epoch': epoch,
                    'train': tr,
                    'val': ev.summary,
                    'perf': {
                        'train_time_s': float(train_time_s),
                        'eval_time_s': float(eval_time_s),
                        'epoch_time_s': float(epoch_time_s),
                        'max_gpu_mem_gb': float(max_gpu_mem_gb),
                    },
                }
            )

            with (run_manager.paths.run_dir / f'threshold_epoch_{epoch:03d}.json').open('w', encoding='utf-8') as f:
                json.dump(ev.threshold_records, f, indent=2)

            logger.info(
                'Epoch %d | train_loss=%.4f (video=%.4f sep=%.4f ov=%.4f fgr=%.4f scale=%.2f) val_loss=%.4f | '
                'val_best_f1=%.4f @thr=%.3f | val_best_acc=%.4f @thr=%.3f | '
                'fg=%.4f val_ov=%.4f | train=%.1fs eval=%.1fs epoch=%.1fs peak_mem=%.2fGB',
                epoch,
                tr['loss'],
                tr['loss_video'],
                tr['loss_sep'],
                tr['loss_overlap'],
                tr['loss_fg_ratio'],
                tr['constraint_scale'],
                ev.summary['val_loss'],
                best_f1['f1'],
                best_f1['threshold'],
                best_acc['acc'],
                best_acc['threshold'],
                ev.summary['fg_ratio_mean'],
                ev.summary.get('mask_overlap_mean', 0.0),
                train_time_s,
                eval_time_s,
                epoch_time_s,
                max_gpu_mem_gb,
            )

            if tb is not None:
                tb.add_scalar('train/loss', tr['loss'], epoch)
                tb.add_scalar('train/loss_video', tr['loss_video'], epoch)
                tb.add_scalar('train/loss_sep', tr['loss_sep'], epoch)
                tb.add_scalar('train/loss_overlap', tr['loss_overlap'], epoch)
                tb.add_scalar('train/loss_fg_ratio', tr['loss_fg_ratio'], epoch)
                tb.add_scalar('train/constraint_scale', tr['constraint_scale'], epoch)
                tb.add_scalar('train/fg_ratio', tr['fg_ratio'], epoch)
                tb.add_scalar('train/mask_overlap', tr['mask_overlap'], epoch)
                tb.add_scalar('train/lr_backbone', lr_backbone, epoch)
                tb.add_scalar('train/lr_head', lr_head, epoch)
                tb.add_scalar('val/loss', ev.summary['val_loss'], epoch)
                tb.add_scalar('val/mask_overlap_mean', ev.summary.get('mask_overlap_mean', 0.0), epoch)
                tb.add_scalar('val_best_f1/f1', best_f1['f1'], epoch)
                tb.add_scalar('val_best_f1/threshold', best_f1['threshold'], epoch)
                tb.add_scalar('val_best_acc/acc', best_acc['acc'], epoch)
                tb.add_scalar('val_best_acc/threshold', best_acc['threshold'], epoch)
                tb.add_scalar('perf/train_time_s', train_time_s, epoch)
                tb.add_scalar('perf/eval_time_s', eval_time_s, epoch)
                tb.add_scalar('perf/epoch_time_s', epoch_time_s, epoch)
                tb.add_scalar('perf/max_gpu_mem_gb', max_gpu_mem_gb, epoch)

        if monitor == 'val_loss':
            improved = monitor_metric < best_metric
        else:
            improved = monitor_metric > best_metric

        if improved:
            best_metric = monitor_metric
            best_threshold = float(monitor_record['threshold'])
            best_metrics = {
                'monitor': monitor_metric,
                'monitor_name': monitor,
                'val_loss': float(ev.summary['val_loss']),
                'f1': float(best_f1['f1']),
                'acc': float(best_acc['acc']),
                'precision': float(monitor_record['precision']),
                'recall': float(monitor_record['recall']),
                'threshold': float(monitor_record['threshold']),
                'best_f1': best_f1,
                'best_acc': best_acc,
            }
            no_improve = 0

            if is_main_process():
                best_ckpt = str(run_manager.paths.ckpt_dir / best_ckpt_name)
                save_checkpoint(
                    path=best_ckpt,
                    model=unwrap(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    extra={
                        'best_threshold': best_threshold,
                        'monitor': monitor,
                        'monitor_metric': best_metric,
                        'best_f1_threshold': float(best_f1['threshold']),
                        'best_acc_threshold': float(best_acc['threshold']),
                    },
                )
        else:
            no_improve += 1

        if is_main_process() and epoch % int(cfg['logging']['save_every_epochs']) == 0:
            save_checkpoint(
                path=str(run_manager.paths.ckpt_dir / 'last.pt'),
                model=unwrap(model),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                extra={'best_threshold': best_threshold, 'monitor': monitor},
            )

        scheduler.step()
        _apply_scheduler_min_lr(optimizer, scheduler)
        barrier()

        if bool(cfg['train']['early_stop'].get('enabled', True)) and no_improve >= patience:
            if is_main_process():
                logger.info('Early stop at epoch=%d (no_improve=%d, monitor=%s)', epoch, no_improve, monitor)
            break

    if is_main_process():
        summary_payload = {
            'monitor': monitor,
            'best_metric': best_metric,
            'best_threshold': best_threshold,
            'best_metrics': best_metrics,
            'best_checkpoint': best_ckpt,
            'last_epoch_metrics': last_epoch_metrics,
        }
        run_manager.dump_summary(summary_payload)
        logger.info('Final metrics JSON:\n%s', json.dumps(summary_payload, ensure_ascii=False, indent=2, default=float))
        if tb is not None:
            tb.flush()
            tb.close()

    return TrainArtifacts(best_metric=best_metric, best_threshold=best_threshold, best_ckpt=best_ckpt)
