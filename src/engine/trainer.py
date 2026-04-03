from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.common.checkpoint import save_checkpoint
from src.common.distributed import barrier, get_world_size, is_main_process
from src.exp.artifact_writer import CsvWriter
from src.exp.run_manager import RunManager

from .evaluator import evaluate

try:
    from tqdm.auto import tqdm
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


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if get_world_size() == 1:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = cfg["train"]["optimizer"]
    name = opt_cfg["name"].lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any]):
    sch_cfg = cfg["train"]["scheduler"]
    name = sch_cfg["name"].lower()
    epochs = int(cfg["train"]["epochs"])

    if name == "cosine":
        warmup = int(sch_cfg.get("warmup_epochs", 0))
        min_lr = float(sch_cfg.get("min_lr", 1e-6))
        base_lr = float(cfg["train"]["optimizer"]["lr"])

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / max(1.0, warmup)
            progress = (epoch - warmup) / max(1, epochs - warmup)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
            return (min_lr / base_lr) + (1 - min_lr / base_lr) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported scheduler: {name}")


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

    accum = int(cfg["train"].get("grad_accum_steps", 1))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
    use_amp = bool(cfg["runtime"].get("use_amp", True))
    amp_dtype = torch.bfloat16 if cfg["runtime"].get("amp_dtype", "bf16") == "bf16" else torch.float16

    running_loss = 0.0
    running_video = 0.0
    running_reg = 0.0
    n_steps = 0
    max_steps = int(cfg["train"].get("max_steps_per_epoch", 0))
    show_progress = bool(cfg["logging"].get("progress_bar", True)) and is_main_process() and tqdm is not None

    optimizer.zero_grad(set_to_none=True)

    iterator = dataloader
    if show_progress:
        total = min(len(dataloader), max_steps) if max_steps > 0 else len(dataloader)
        iterator = tqdm(dataloader, total=total, desc=f"Train E{epoch:03d}", leave=False)

    for step, batch in enumerate(iterator):
        windows = batch["windows"].to(device, non_blocking=True)
        window_valid = batch["window_valid"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
            out = model(windows, window_valid)
            loss_dict = criterion(out["video_logit"], label, out["window_probs"])
            loss = loss_dict["loss"] / accum

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

        running_loss += float(loss_dict["loss"].detach().item())
        running_video += float(loss_dict["loss_video"].detach().item())
        running_reg += float(loss_dict["loss_reg"].detach().item())
        n_steps += 1

        if is_main_process() and step % int(cfg["logging"]["log_interval"]) == 0:
            logger.info(
                "Epoch %d Step %d/%d | loss=%.4f video=%.4f reg=%.4f",
                epoch,
                step,
                len(dataloader),
                running_loss / n_steps,
                running_video / n_steps,
                running_reg / n_steps,
            )

        if max_steps > 0 and (step + 1) >= max_steps:
            break

    # Flush remaining gradients when steps are not divisible by accum.
    if n_steps % accum != 0:
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

    total = torch.tensor([running_loss, running_video, running_reg, n_steps], dtype=torch.float64, device=device)
    total = _all_reduce_sum(total)

    denom = max(float(total[3].item()), 1.0)
    return {
        "loss": float(total[0].item() / denom),
        "loss_video": float(total[1].item() / denom),
        "loss_reg": float(total[2].item() / denom),
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
    use_amp = bool(cfg["runtime"].get("use_amp", True))
    amp_dtype_name = cfg["runtime"].get("amp_dtype", "bf16")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16 and device.type == "cuda")

    history_writer = CsvWriter(
        path=str(run_manager.paths.run_dir / "history.csv"),
        headers=[
            "epoch",
            "train_loss",
            "train_loss_video",
            "train_loss_reg",
            "val_acc_fixed",
            "val_precision_fixed",
            "val_recall_fixed",
            "val_f1_fixed",
            "val_acc_best",
            "val_precision_best",
            "val_recall_best",
            "val_f1_best",
            "best_threshold",
            "val_acc_scan_acc",
            "val_acc_scan_f1",
            "val_acc_scan_threshold",
            "lr",
            "acg_tau",
            "acg_temp",
        ],
    )
    tb_writer = None
    if is_main_process() and bool(cfg["logging"].get("tensorboard", True)) and SummaryWriter is not None:
        tb_dir = str(run_manager.paths.run_dir / "tensorboard")
        tb_writer = SummaryWriter(log_dir=tb_dir)

    best_metric = -1.0
    best_threshold = float(cfg["eval"]["threshold"]["fixed"])
    best_ckpt = ""
    best_metrics_record: dict[str, float] | None = None
    best_metric_acc = -1.0
    best_threshold_acc = float(cfg["eval"]["threshold"]["fixed"])
    best_ckpt_acc = ""
    best_metrics_acc_record: dict[str, float] | None = None
    best_monitor = -1.0
    monitor_name = str(cfg["train"]["early_stop"].get("monitor", "f1")).lower()
    if monitor_name not in {"f1", "acc"}:
        monitor_name = "f1"
    no_improve = 0

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["early_stop"].get("patience", 10))

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch,
            logger=logger,
        )

        eval_result = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            fixed_threshold=float(cfg["eval"]["threshold"]["fixed"]),
            scan_cfg=cfg["eval"]["threshold"],
            max_batches=int(cfg["eval"].get("max_batches", 0)),
            show_progress=bool(cfg["logging"].get("progress_bar", True)),
        )

        fixed = eval_result.summary["fixed"]
        best = eval_result.summary["best"]
        best_acc_scan = max(
            eval_result.threshold_records,
            key=lambda x: (float(x["acc"]), float(x["f1"])),
        )

        if is_main_process():
            acg_state = _unwrap(model).acg_state()
            lr = optimizer.param_groups[0]["lr"]

            history_writer.write(
                {
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "train_loss_video": train_stats["loss_video"],
                    "train_loss_reg": train_stats["loss_reg"],
                    "val_acc_fixed": fixed["acc"],
                    "val_precision_fixed": fixed["precision"],
                    "val_recall_fixed": fixed["recall"],
                    "val_f1_fixed": fixed["f1"],
                    "val_acc_best": best["acc"],
                    "val_precision_best": best["precision"],
                    "val_recall_best": best["recall"],
                    "val_f1_best": best["f1"],
                    "best_threshold": best["threshold"],
                    "val_acc_scan_acc": best_acc_scan["acc"],
                    "val_acc_scan_f1": best_acc_scan["f1"],
                    "val_acc_scan_threshold": best_acc_scan["threshold"],
                    "lr": lr,
                    "acg_tau": acg_state["tau"],
                    "acg_temp": acg_state["temp"],
                }
            )

            run_manager.append_metrics(
                {
                    "epoch": epoch,
                    "train": train_stats,
                    "val": eval_result.summary,
                }
            )

            thr_path = run_manager.paths.run_dir / f"threshold_epoch_{epoch:03d}.json"
            with thr_path.open("w", encoding="utf-8") as f:
                json.dump(eval_result.threshold_records, f, indent=2)

            logger.info(
                (
                    "Epoch %d | train_loss=%.4f | "
                    "val_best_f1: f1=%.4f p=%.4f r=%.4f acc=%.4f @thr=%.3f | "
                    "val_best_acc: acc=%.4f f1=%.4f @thr=%.3f"
                ),
                epoch,
                train_stats["loss"],
                best["f1"],
                best["precision"],
                best["recall"],
                best["acc"],
                best["threshold"],
                best_acc_scan["acc"],
                best_acc_scan["f1"],
                best_acc_scan["threshold"],
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", train_stats["loss"], epoch)
                tb_writer.add_scalar("train/loss_video", train_stats["loss_video"], epoch)
                tb_writer.add_scalar("train/loss_reg", train_stats["loss_reg"], epoch)
                tb_writer.add_scalar("train/lr", lr, epoch)
                tb_writer.add_scalar("model/acg_tau", acg_state["tau"], epoch)
                tb_writer.add_scalar("model/acg_temp", acg_state["temp"], epoch)

                tb_writer.add_scalar("val_fixed/acc", fixed["acc"], epoch)
                tb_writer.add_scalar("val_fixed/precision", fixed["precision"], epoch)
                tb_writer.add_scalar("val_fixed/recall", fixed["recall"], epoch)
                tb_writer.add_scalar("val_fixed/f1", fixed["f1"], epoch)

                tb_writer.add_scalar("val_best/acc", best["acc"], epoch)
                tb_writer.add_scalar("val_best/precision", best["precision"], epoch)
                tb_writer.add_scalar("val_best/recall", best["recall"], epoch)
                tb_writer.add_scalar("val_best/f1", best["f1"], epoch)
                tb_writer.add_scalar("val_best/threshold", best["threshold"], epoch)
                tb_writer.add_scalar("val_acc_scan/acc", best_acc_scan["acc"], epoch)
                tb_writer.add_scalar("val_acc_scan/f1", best_acc_scan["f1"], epoch)
                tb_writer.add_scalar("val_acc_scan/threshold", best_acc_scan["threshold"], epoch)

        metric_f1 = float(best["f1"])
        improved_f1 = metric_f1 > best_metric

        if improved_f1:
            best_metric = metric_f1
            best_threshold = float(best["threshold"])
            best_metrics_record = {k: float(v) for k, v in best.items() if isinstance(v, (int, float))}
            if is_main_process():
                best_ckpt = str(run_manager.paths.ckpt_dir / "best_f1.pt")
                save_checkpoint(
                    ckpt_path=best_ckpt,
                    model=_unwrap(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    extra={"best_threshold": best_threshold},
                )

        metric_acc = float(best_acc_scan["acc"])
        improved_acc = metric_acc > best_metric_acc
        if improved_acc:
            best_metric_acc = metric_acc
            best_threshold_acc = float(best_acc_scan["threshold"])
            best_metrics_acc_record = {
                k: float(v)
                for k, v in best_acc_scan.items()
                if isinstance(v, (int, float))
            }
            if is_main_process():
                best_ckpt_acc = str(run_manager.paths.ckpt_dir / "best_acc.pt")
                save_checkpoint(
                    ckpt_path=best_ckpt_acc,
                    model=_unwrap(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric_acc,
                    extra={"best_threshold": best_threshold_acc},
                )

        monitor_value = metric_f1 if monitor_name == "f1" else metric_acc
        if monitor_value > best_monitor:
            best_monitor = monitor_value
            no_improve = 0
        else:
            no_improve += 1

        if is_main_process() and epoch % int(cfg["logging"]["save_every_epochs"]) == 0:
            last_ckpt = str(run_manager.paths.ckpt_dir / "last.pt")
            save_checkpoint(
                ckpt_path=last_ckpt,
                model=_unwrap(model),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                extra={"best_threshold": best_threshold},
            )

        scheduler.step()
        barrier()

        if bool(cfg["train"]["early_stop"].get("enabled", True)) and no_improve >= patience:
            if is_main_process():
                logger.info("Early stopping at epoch=%d, no_improve=%d", epoch, no_improve)
            break

    if is_main_process():
        run_manager.dump_summary(
            {
                "best_metric_f1": best_metric,
                "best_threshold": best_threshold,
                "best_metrics": best_metrics_record or {},
                "best_checkpoint": best_ckpt,
                "best_metric_acc": best_metric_acc,
                "best_threshold_acc": best_threshold_acc,
                "best_metrics_acc": best_metrics_acc_record or {},
                "best_checkpoint_acc": best_ckpt_acc,
                "early_stop_monitor": monitor_name,
                "best_monitor_value": best_monitor,
            }
        )
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    return TrainArtifacts(best_metric=best_metric, best_threshold=best_threshold, best_ckpt=best_ckpt)
