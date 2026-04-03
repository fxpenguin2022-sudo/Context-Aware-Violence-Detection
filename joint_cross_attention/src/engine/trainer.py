from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scene_decoupling.src.common.checkpoint import save_checkpoint
from scene_decoupling.src.common.distributed import all_reduce_sum, barrier, is_main_process
from scene_decoupling.src.exp.artifact_writer import CsvWriter
from scene_decoupling.src.exp.run_manager import RunManager

from joint_cross_attention.src.engine.evaluator import evaluate

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


def _dump_eval_artifacts(run_dir, epoch: int, ev: Any) -> None:
    payload = {
        'epoch': int(epoch),
        'summary': ev.summary,
        'predictions': ev.predictions,
        'threshold_scan': ev.threshold_records,
    }
    with (run_dir / f'val_predictions_epoch_{epoch:03d}.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _dump_best_eval_artifacts(run_dir, tag: str, epoch: int, ev: Any) -> None:
    payload = {
        'epoch': int(epoch),
        'summary': ev.summary,
        'predictions': ev.predictions,
        'threshold_scan': ev.threshold_records,
    }
    with (run_dir / f'{tag}.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _split_param_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    branch_params: list[torch.nn.Parameter] = []
    fusion_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('pose_branch.') or name.startswith('context_branch.') or name.startswith('module.pose_branch.') or name.startswith('module.context_branch.'):
            branch_params.append(param)
        else:
            fusion_params.append(param)
    return branch_params, fusion_params


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = cfg['train']['optimizer']
    if str(opt_cfg['name']).lower() != 'adamw':
        raise ValueError(f"Unsupported optimizer {opt_cfg['name']}")
    branch_params, fusion_params = _split_param_groups(model)
    groups = []
    if branch_params:
        groups.append({'params': branch_params, 'lr': float(opt_cfg['lr_branch']), 'name': 'branch'})
    if fusion_params:
        groups.append({'params': fusion_params, 'lr': float(opt_cfg['lr_fusion']), 'name': 'fusion'})
    return torch.optim.AdamW(groups, weight_decay=float(opt_cfg['weight_decay']), betas=tuple(opt_cfg.get('betas', [0.9, 0.999])))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any]):
    sch_cfg = cfg['train']['scheduler']
    epochs = int(cfg['train']['epochs'])
    warmup = int(sch_cfg.get('warmup_epochs', 0))
    min_lr = float(sch_cfg.get('min_lr', 1e-6))
    base_lrs = [max(pg['lr'], 1e-12) for pg in optimizer.param_groups]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())
        return cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.min_lr = min_lr  # type: ignore[attr-defined]
    scheduler.base_lrs = base_lrs  # type: ignore[attr-defined]
    return scheduler


def _apply_scheduler_min_lr(optimizer: torch.optim.Optimizer, scheduler) -> None:
    min_lr = float(getattr(scheduler, 'min_lr', 0.0))
    for pg in optimizer.param_groups:
        pg['lr'] = max(float(pg['lr']), min_lr)


def _normalize_monitor(monitor: str) -> str:
    key = str(monitor).strip().lower()
    aliases = {'loss': 'val_loss', 'val_loss': 'val_loss', 'f1': 'f1', 'val_f1': 'f1', 'acc': 'acc', 'accuracy': 'acc', 'val_acc': 'acc'}
    if key not in aliases:
        raise ValueError(f'Unsupported early-stop monitor: {monitor}')
    return aliases[key]


def _select_monitor_record(monitor: str, ev_summary: dict[str, Any]) -> tuple[float, dict[str, float]]:
    best_f1 = ev_summary.get('best_f1', ev_summary['best'])
    best_acc = ev_summary.get('best_acc', best_f1)
    if monitor == 'val_loss':
        return float(ev_summary['val_loss']), best_f1
    if monitor == 'f1':
        return float(best_f1['f1']), best_f1
    if monitor == 'acc':
        return float(best_acc['acc']), best_acc
    raise ValueError(f'Unsupported monitor: {monitor}')


def _monitor_checkpoint_name(monitor: str) -> str:
    return 'best_val_loss.pt' if monitor == 'val_loss' else f'best_{monitor}.pt'


def _distributed_max_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _progress_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get('experiment', {}).get('name', 'joint_model'))


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
    logger,
) -> dict[str, float]:
    model.train()
    accum = int(cfg['train'].get('grad_accum_steps', 1))
    grad_clip = float(cfg['train'].get('grad_clip_norm', 0.0))
    use_amp = bool(cfg['runtime'].get('use_amp', True))
    amp_dtype = torch.bfloat16 if cfg['runtime'].get('amp_dtype', 'bf16') == 'bf16' else torch.float16
    max_steps = int(cfg['train'].get('max_steps_per_epoch', 0))

    show_pbar = bool(cfg['logging'].get('progress_bar', True)) and is_main_process() and tqdm is not None
    iterator = dataloader
    if show_pbar:
        total = min(len(dataloader), max_steps) if max_steps > 0 else len(dataloader)
        iterator = tqdm(dataloader, total=total, desc=f'{_progress_name(cfg)} Train E{epoch:03d}', leave=False, dynamic_ncols=True, miniters=1, file=sys.stdout)

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_alpha = 0.0
    running_beta = 0.0
    running_gamma = 0.0
    steps = 0

    for step, batch in enumerate(iterator):
        pose_windows = batch['pose_windows'].to(device, non_blocking=True)
        pose_window_valid = batch['pose_window_valid'].to(device, non_blocking=True)
        video_clips = batch['video_clips'].to(device, non_blocking=True)
        video_poses = batch['video_poses'].to(device, non_blocking=True)
        clip_valid_mask = batch['clip_valid_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
            out = model(pose_windows, pose_window_valid, video_clips, video_poses, clip_valid_mask)
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
        running_alpha += float(out['alpha'].detach().mean().item())
        running_beta += float(out['beta'].detach().mean().item())
        running_gamma += float(out['gamma'].detach().mean().item())
        steps += 1

        if show_pbar:
            iterator.set_postfix({'loss': f'{running_loss/max(1,steps):.4f}', 'a': f'{running_alpha/max(1,steps):.3f}', 'b': f'{running_beta/max(1,steps):.3f}', 'g': f'{running_gamma/max(1,steps):.3f}'}, refresh=True)

        if is_main_process() and step % int(cfg['logging']['log_interval']) == 0 and not show_pbar:
            logger.info('Epoch %d Step %d/%d | loss=%.4f alpha_mean=%.3f beta_mean=%.3f gamma_mean=%.3f', epoch, step, len(dataloader), running_loss/max(1,steps), running_alpha/max(1,steps), running_beta/max(1,steps), running_gamma/max(1,steps))

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

    stat = torch.tensor([running_loss, running_alpha, running_beta, running_gamma, steps], dtype=torch.float64, device=device)
    stat = all_reduce_sum(stat)
    denom = max(float(stat[4].item()), 1.0)
    return {
        'loss': float(stat[0].item() / denom),
        'alpha_mean': float(stat[1].item() / denom),
        'beta_mean': float(stat[2].item() / denom),
        'gamma_mean': float(stat[3].item() / denom),
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
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16) if device.type == 'cuda' else None

    history = CsvWriter(
        path=str(run_manager.paths.run_dir / 'history.csv'),
        headers=['epoch', 'train_loss', 'train_alpha_mean', 'train_beta_mean', 'train_gamma_mean', 'val_loss', 'val_acc_fixed', 'val_f1_fixed', 'val_acc_best', 'val_f1_best', 'best_threshold', 'val_alpha_mean', 'val_beta_mean', 'val_gamma_mean', 'fg_ratio_mean', 'mask_overlap_mean', 'lr_branch', 'lr_fusion', 'train_time_s', 'eval_time_s', 'epoch_time_s', 'max_gpu_mem_gb'],
    )

    tb = None
    if is_main_process() and bool(cfg['logging'].get('tensorboard', True)) and SummaryWriter is not None:
        tb = SummaryWriter(log_dir=str(run_manager.paths.run_dir / 'tensorboard'))

    monitor = _normalize_monitor(str(cfg['train']['early_stop'].get('monitor', 'acc')))
    best_metric = float('inf') if monitor == 'val_loss' else -1.0
    best_threshold = float(cfg['eval']['threshold']['fixed'])
    best_ckpt = ''
    best_ckpt_name = _monitor_checkpoint_name(monitor)
    best_metrics: dict[str, Any] = {}
    no_improve = 0
    epochs = int(cfg['train']['epochs'])
    patience = int(cfg['train']['early_stop'].get('patience', 10))

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        epoch_t0 = time.perf_counter()
        train_t0 = time.perf_counter()
        tr = train_one_epoch(model, criterion, train_loader, optimizer, scaler, device, cfg, epoch, logger)
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
            max_batches=int(cfg['eval'].get('max_batches', 0)),
            show_progress=bool(cfg['logging'].get('progress_bar', True)),
            progress_name=_progress_name(cfg),
        )
        eval_time_s = time.perf_counter() - eval_t0
        epoch_time_s = time.perf_counter() - epoch_t0
        max_gpu_mem_local = float(torch.cuda.max_memory_allocated(device) / (1024.0**3)) if device.type == 'cuda' else 0.0
        max_gpu_mem_gb = _distributed_max_scalar(max_gpu_mem_local, device=device)

        fixed = ev.summary['fixed']
        best_f1 = ev.summary['best_f1']
        best_acc = ev.summary['best_acc']
        monitor_metric, monitor_record = _select_monitor_record(monitor, ev.summary)
        lr_branch = float(optimizer.param_groups[0]['lr'])
        lr_fusion = float(optimizer.param_groups[-1]['lr'])

        if is_main_process():
            history.write(
                {
                    'epoch': epoch,
                    'train_loss': tr['loss'],
                    'train_alpha_mean': tr['alpha_mean'],
                    'train_beta_mean': tr['beta_mean'],
                    'train_gamma_mean': tr['gamma_mean'],
                    'val_loss': ev.summary['val_loss'],
                    'val_acc_fixed': fixed['acc'],
                    'val_f1_fixed': fixed['f1'],
                    'val_acc_best': best_acc['acc'],
                    'val_f1_best': best_acc['f1'],
                    'best_threshold': best_acc['threshold'],
                    'val_alpha_mean': ev.summary['alpha_mean'],
                    'val_beta_mean': ev.summary['beta_mean'],
                    'val_gamma_mean': ev.summary['gamma_mean'],
                    'fg_ratio_mean': ev.summary['fg_ratio_mean'],
                    'mask_overlap_mean': ev.summary['mask_overlap_mean'],
                    'lr_branch': lr_branch,
                    'lr_fusion': lr_fusion,
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
                    'perf': {'train_time_s': train_time_s, 'eval_time_s': eval_time_s, 'epoch_time_s': epoch_time_s, 'max_gpu_mem_gb': max_gpu_mem_gb},
                }
            )
            with (run_manager.paths.run_dir / f'threshold_epoch_{epoch:03d}.json').open('w', encoding='utf-8') as f:
                json.dump(ev.threshold_records, f, indent=2)
            _dump_eval_artifacts(run_manager.paths.run_dir, epoch, ev)
            logger.info('Epoch %d | train_loss=%.4f val_loss=%.4f | val_best_acc=%.4f @thr=%.3f | alpha_mean=%.3f beta_mean=%.3f gamma_mean=%.3f | train=%.1fs eval=%.1fs epoch=%.1fs peak_mem=%.2fGB', epoch, tr['loss'], ev.summary['val_loss'], best_acc['acc'], best_acc['threshold'], ev.summary['alpha_mean'], ev.summary['beta_mean'], ev.summary['gamma_mean'], train_time_s, eval_time_s, epoch_time_s, max_gpu_mem_gb)
            if tb is not None:
                tb.add_scalar('train/loss', tr['loss'], epoch)
                tb.add_scalar('train/alpha_mean', tr['alpha_mean'], epoch)
                tb.add_scalar('train/beta_mean', tr['beta_mean'], epoch)
                tb.add_scalar('train/gamma_mean', tr['gamma_mean'], epoch)
                tb.add_scalar('val/loss', ev.summary['val_loss'], epoch)
                tb.add_scalar('val/alpha_mean', ev.summary['alpha_mean'], epoch)
                tb.add_scalar('val/beta_mean', ev.summary['beta_mean'], epoch)
                tb.add_scalar('val/gamma_mean', ev.summary['gamma_mean'], epoch)
                tb.add_scalar('val_best_acc/acc', best_acc['acc'], epoch)
                tb.add_scalar('val_best_acc/threshold', best_acc['threshold'], epoch)
                tb.add_scalar('perf/train_time_s', train_time_s, epoch)
                tb.add_scalar('perf/eval_time_s', eval_time_s, epoch)
                tb.add_scalar('perf/epoch_time_s', epoch_time_s, epoch)
                tb.add_scalar('perf/max_gpu_mem_gb', max_gpu_mem_gb, epoch)

        improved = monitor_metric < best_metric if monitor == 'val_loss' else monitor_metric > best_metric
        if improved:
            best_metric = monitor_metric
            best_threshold = float(monitor_record['threshold'])
            best_metrics = {
                'monitor': monitor_metric,
                'monitor_name': monitor,
                'val_loss': float(ev.summary['val_loss']),
                'acc': float(best_acc['acc']),
                'f1': float(best_f1['f1']),
                'precision': float(monitor_record['precision']),
                'recall': float(monitor_record['recall']),
                'threshold': float(monitor_record['threshold']),
                'alpha_mean': float(ev.summary['alpha_mean']),
                'beta_mean': float(ev.summary['beta_mean']),
                'gamma_mean': float(ev.summary['gamma_mean']),
                'best_f1': best_f1,
                'best_acc': best_acc,
            }
            no_improve = 0
            if is_main_process():
                best_ckpt = str(run_manager.paths.ckpt_dir / best_ckpt_name)
                save_checkpoint(best_ckpt, unwrap(model), optimizer, scheduler, scaler, epoch, best_metric, extra={'best_threshold': best_threshold, 'monitor': monitor, 'monitor_metric': best_metric})
                _dump_best_eval_artifacts(run_manager.paths.run_dir, f'best_{monitor}_eval', epoch, ev)
                _dump_best_eval_artifacts(run_manager.paths.run_dir, 'best_acc_eval', epoch, ev)
        else:
            no_improve += 1

        if is_main_process() and epoch % int(cfg['logging'].get('save_every_epochs', 1)) == 0:
            save_checkpoint(str(run_manager.paths.ckpt_dir / 'last.pt'), unwrap(model), optimizer, scheduler, scaler, epoch, best_metric, extra={'best_threshold': best_threshold, 'monitor': monitor})

        scheduler.step()
        _apply_scheduler_min_lr(optimizer, scheduler)
        barrier()

        if bool(cfg['train']['early_stop'].get('enabled', True)) and no_improve >= patience:
            if is_main_process():
                logger.info('Early stop at epoch=%d (no_improve=%d, monitor=%s)', epoch, no_improve, monitor)
            break

    if is_main_process():
        summary_payload = {'monitor': monitor, 'best_metric': best_metric, 'best_threshold': best_threshold, 'best_metrics': best_metrics, 'best_checkpoint': best_ckpt}
        run_manager.dump_summary(summary_payload)
        logger.info('Final metrics JSON:\n%s', json.dumps(summary_payload, ensure_ascii=False, indent=2, default=float))
        if tb is not None:
            tb.flush()
            tb.close()

    return TrainArtifacts(best_metric=best_metric, best_threshold=best_threshold, best_ckpt=best_ckpt)
