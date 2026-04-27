import argparse
import math
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stnn_bench import (
    StateNormalizer,
    TrajectoryTensorDataset,
    build_model,
    build_system,
    compute_rollout_metrics,
    flatten_transitions,
    load_config,
    load_or_generate_dataset,
    plot_rollout_samples,
    save_config_snapshot,
    set_global_seed,
)
from stnn_bench.config_utils import dump_metrics, resolve_run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StNN and baselines on nonlinear systems.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument("--notes", type=str, default=None, help="Optional freeform notes.")
    return parser.parse_args()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _teacher_forcing_ratio(
    epoch: int,
    epochs: int,
    ratio_start: float,
    ratio_end: float,
) -> float:
    if epochs <= 1:
        return ratio_end
    alpha = epoch / float(epochs - 1)
    return (1.0 - alpha) * ratio_start + alpha * ratio_end


def compute_one_step_mse(
    *,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    criterion = nn.MSELoss(reduction="mean")
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch_x = x[start : start + batch_size].to(device)
            batch_y = y[start : start + batch_size].to(device)
            pred = model(batch_x)
            losses.append(criterion(pred, batch_y).item())
    return float(np.mean(losses)) if losses else float("nan")


def train_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
    rollout_steps_cfg: int,
    one_step_weight: float,
    rollout_weight: float,
    reg_weight: float,
    grad_clip_norm: float,
    tf_start: float,
    tf_end: float,
) -> Dict[str, float]:
    criterion = nn.MSELoss(reduction="mean")
    model.train()

    teacher_ratio = _teacher_forcing_ratio(epoch, epochs, tf_start, tf_end)

    total_loss = 0.0
    total_one = 0.0
    total_roll = 0.0
    total_reg = 0.0
    total_batches = 0

    for trajectories in loader:
        trajectories = trajectories.to(device)
        batch_size, seq_len, _ = trajectories.shape

        if seq_len < 3:
            continue

        rollout_steps = min(rollout_steps_cfg, seq_len - 1)
        if rollout_steps < 1:
            continue

        max_start = seq_len - rollout_steps - 1
        if max_start < 0:
            continue

        batch_index = torch.arange(batch_size, device=device)
        start_idx = torch.randint(0, max_start + 1, (batch_size,), device=device)

        x0 = trajectories[batch_index, start_idx]
        target1 = trajectories[batch_index, start_idx + 1]

        pred1 = model(x0)
        one_step_loss = criterion(pred1, target1)

        rollout_loss = torch.tensor(0.0, device=device)
        current = x0
        for step in range(1, rollout_steps + 1):
            target = trajectories[batch_index, start_idx + step]
            pred = model(current)
            rollout_loss = rollout_loss + criterion(pred, target)

            if step < rollout_steps:
                if teacher_ratio <= 0.0:
                    current = pred
                elif teacher_ratio >= 1.0:
                    current = target
                else:
                    mask = (torch.rand(batch_size, 1, device=device) < teacher_ratio).float()
                    current = mask * target + (1.0 - mask) * pred

        rollout_loss = rollout_loss / float(rollout_steps)
        reg_loss = model.regularization_loss()

        loss = (
            one_step_weight * one_step_loss
            + rollout_weight * rollout_loss
            + reg_weight * reg_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_one += one_step_loss.item()
        total_roll += rollout_loss.item()
        total_reg += reg_loss.item()
        total_batches += 1

    if total_batches == 0:
        return {
            "train_loss": float("nan"),
            "train_one_step": float("nan"),
            "train_rollout": float("nan"),
            "train_reg": float("nan"),
            "teacher_forcing_ratio": teacher_ratio,
        }

    denom = float(total_batches)
    return {
        "train_loss": total_loss / denom,
        "train_one_step": total_one / denom,
        "train_rollout": total_roll / denom,
        "train_reg": total_reg / denom,
        "teacher_forcing_ratio": teacher_ratio,
    }


def save_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    normalizer: StateNormalizer,
    best_metric: float,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "normalizer_mean": normalizer.mean,
        "normalizer_std": normalizer.std,
        "best_metric": best_metric,
    }
    torch.save(payload, path)


def _load_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", float("inf")))
    mean = checkpoint["normalizer_mean"].astype(np.float32)
    std = checkpoint["normalizer_std"].astype(np.float32)
    return epoch, best_metric, mean, std


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.notes:
        config.setdefault("experiment", {})["notes"] = args.notes

    base_seed = int(args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 42))
    set_global_seed(base_seed)

    run_dir, run_name = resolve_run_directory(config)
    save_config_snapshot(config, run_dir)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        requested = str(config.get("experiment", {}).get("device", "cuda")).lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    system = build_system(config.get("system", {}))
    dataset_cfg = dict(config.get("dataset", {}))
    data_bundle = load_or_generate_dataset(
        system=system,
        dataset_cfg=dataset_cfg,
        seed=base_seed,
        run_dir=run_dir,
    )

    normalizer = StateNormalizer.from_trajectories(data_bundle.train)
    train_norm = normalizer.transform_np(data_bundle.train)
    val_norm = normalizer.transform_np(data_bundle.val)
    test_norm = normalizer.transform_np(data_bundle.test)

    train_dataset = TrajectoryTensorDataset(torch.from_numpy(train_norm))

    training_cfg = dict(config.get("training", {}))
    batch_size = int(training_cfg.get("batch_size", 64))
    epochs = int(training_cfg.get("epochs", 100))
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    warmup_fraction = float(training_cfg.get("warmup_fraction", 0.05))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))
    rollout_steps = int(training_cfg.get("rollout_steps", 8))
    rollout_loss_weight = float(training_cfg.get("rollout_loss_weight", 0.5))
    one_step_loss_weight = float(training_cfg.get("one_step_loss_weight", 1.0))
    reg_loss_weight = float(training_cfg.get("reg_loss_weight", 1e-3))
    tf_start = float(training_cfg.get("teacher_forcing_start", 1.0))
    tf_end = float(training_cfg.get("teacher_forcing_end", 0.2))
    eval_every = int(training_cfg.get("eval_every", 5))
    val_batch_size = int(training_cfg.get("val_batch_size", 4096))
    checkpoint_metric = str(training_cfg.get("checkpoint_metric", "rollout_rmse_h100"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )

    dt = float(dataset_cfg.get("dt", 0.1))
    model = build_model(config.get("model", {}), state_dim=system.state_dim, dt=dt).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * max(1, len(train_loader)))
    warmup_steps = int(warmup_fraction * total_steps)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    start_epoch = 0
    best_metric = float("inf")

    if args.resume:
        start_epoch, best_metric, ckpt_mean, ckpt_std = _load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = start_epoch + 1
        normalizer = StateNormalizer(mean=ckpt_mean, std=ckpt_std)
        train_norm = normalizer.transform_np(data_bundle.train)
        val_norm = normalizer.transform_np(data_bundle.val)
        test_norm = normalizer.transform_np(data_bundle.test)

        train_dataset = TrajectoryTensorDataset(torch.from_numpy(train_norm))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=int(training_cfg.get("num_workers", 0)),
        )

    print(f"Run directory: {run_dir}")
    print(f"Run name: {run_name}")
    print(f"Device: {device}")
    print(f"System: {system.name} | state_dim={system.state_dim}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    val_x_np, val_y_np = flatten_transitions(val_norm)
    val_x = torch.from_numpy(val_x_np)
    val_y = torch.from_numpy(val_y_np)

    test_x_np, test_y_np = flatten_transitions(test_norm)
    test_x = torch.from_numpy(test_x_np)
    test_y = torch.from_numpy(test_y_np)

    eval_cfg = dict(config.get("evaluation", {}))
    horizons = [int(v) for v in eval_cfg.get("rollout_horizons", [20, 50, 100])]
    num_rollout_samples = int(eval_cfg.get("num_rollout_samples", 64))
    divergence_threshold = float(eval_cfg.get("divergence_threshold", 1.5))

    history_path = os.path.join(run_dir, "history.jsonl")
    best_ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
    last_ckpt_path = os.path.join(run_dir, "checkpoint_last.pt")

    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        train_metrics = train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=train_loader,
            device=device,
            epoch=epoch,
            epochs=epochs,
            rollout_steps_cfg=rollout_steps,
            one_step_weight=one_step_loss_weight,
            rollout_weight=rollout_loss_weight,
            reg_weight=reg_loss_weight,
            grad_clip_norm=grad_clip_norm,
            tf_start=tf_start,
            tf_end=tf_end,
        )

        epoch_metrics: Dict[str, float] = {
            "epoch": float(epoch + 1),
            **train_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        should_eval = (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs
        if should_eval:
            val_one_step = compute_one_step_mse(
                model=model,
                x=val_x,
                y=val_y,
                device=device,
                batch_size=val_batch_size,
            )
            rollout_metrics = compute_rollout_metrics(
                model=model,
                trajectories_norm=val_norm,
                normalizer=normalizer,
                device=device,
                horizons=horizons,
                num_samples=num_rollout_samples,
                observable_indices=system.observable_indices,
                divergence_threshold=divergence_threshold,
                rng_seed=base_seed + epoch,
            )
            epoch_metrics["val_one_step_mse"] = val_one_step
            epoch_metrics.update({f"val_{k}": v for k, v in rollout_metrics.items()})

            target_metric = epoch_metrics.get(f"val_{checkpoint_metric}", val_one_step)
            if target_metric < best_metric:
                best_metric = target_metric
                save_checkpoint(
                    path=best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    normalizer=normalizer,
                    best_metric=best_metric,
                )

            save_checkpoint(
                path=last_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                normalizer=normalizer,
                best_metric=best_metric,
            )

        with open(history_path, "a", encoding="utf-8") as history_file:
            history_file.write(str(epoch_metrics).replace("'", '"') + "\n")

        message = (
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train: {train_metrics['train_loss']:.6f} | "
            f"One-step: {train_metrics['train_one_step']:.6f}"
        )
        if "val_one_step_mse" in epoch_metrics:
            message += f" | Val one-step: {epoch_metrics['val_one_step_mse']:.6f}"
            for horizon in horizons:
                key = f"val_rollout_rmse_h{horizon}"
                if key in epoch_metrics:
                    message += f" | RMSE@{horizon}: {epoch_metrics[key]:.4f}"
        print(message)

    # Load best model for final test metrics.
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    test_one_step = compute_one_step_mse(
        model=model,
        x=test_x,
        y=test_y,
        device=device,
        batch_size=val_batch_size,
    )
    test_rollout = compute_rollout_metrics(
        model=model,
        trajectories_norm=test_norm,
        normalizer=normalizer,
        device=device,
        horizons=horizons,
        num_samples=int(eval_cfg.get("test_rollout_samples", num_rollout_samples)),
        observable_indices=system.observable_indices,
        divergence_threshold=divergence_threshold,
        rng_seed=base_seed + 999,
    )

    final_metrics = {
        "test_one_step_mse": test_one_step,
        **{f"test_{k}": v for k, v in test_rollout.items()},
        "wall_time_seconds": time.time() - start_time,
        "trainable_parameters": float(count_parameters(model)),
    }
    dump_metrics(final_metrics, run_dir, filename="test_metrics.json")

    if bool(eval_cfg.get("save_plots", True)):
        plot_rollout_samples(
            model=model,
            trajectories_norm=test_norm,
            normalizer=normalizer,
            t_eval=data_bundle.t_eval,
            state_labels=system.state_labels,
            observable_indices=system.observable_indices,
            out_dir=os.path.join(run_dir, "plots"),
            num_samples=int(eval_cfg.get("plot_samples", 5)),
            rng_seed=base_seed + 1234,
            device=device,
            prefix="test",
        )

    print("Training complete.")
    print(f"Best validation metric ({checkpoint_metric}): {best_metric:.6f}")
    print(f"Test one-step MSE: {test_one_step:.6f}")
    for horizon in horizons:
        key = f"rollout_rmse_h{horizon}"
        test_key = f"test_{key}"
        if test_key in final_metrics:
            print(f"Test RMSE@{horizon}: {final_metrics[test_key]:.6f}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
