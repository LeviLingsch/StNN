from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import StateNormalizer


def rollout_model(
    model: torch.nn.Module,
    init_state: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    preds = [init_state]
    current = init_state
    for _ in range(horizon):
        current = model(current)
        preds.append(current)
    return torch.stack(preds, dim=1)


def _first_divergence_step(errors: np.ndarray, threshold: float) -> int:
    indices = np.where(errors > threshold)[0]
    if indices.size == 0:
        return errors.shape[0] - 1
    return int(indices[0])


def compute_rollout_metrics(
    *,
    model: torch.nn.Module,
    trajectories_norm: np.ndarray,
    normalizer: StateNormalizer,
    device: torch.device,
    horizons: Sequence[int],
    num_samples: int,
    observable_indices: Sequence[int],
    divergence_threshold: float,
    rng_seed: int,
) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(rng_seed)

    num_traj = trajectories_norm.shape[0]
    sample_count = min(num_samples, num_traj)
    sampled_indices = rng.choice(num_traj, size=sample_count, replace=False)

    max_horizon = min(max(horizons), trajectories_norm.shape[1] - 1)
    obs = np.array(list(observable_indices), dtype=np.int64)

    all_pred = []
    all_true = []

    with torch.no_grad():
        for idx in sampled_indices:
            traj_norm = trajectories_norm[idx]
            init = torch.tensor(traj_norm[0], dtype=torch.float32, device=device).unsqueeze(0)
            pred_norm = rollout_model(model, init, max_horizon).squeeze(0).cpu().numpy()
            true_norm = traj_norm[: max_horizon + 1]

            all_pred.append(normalizer.inverse_np(pred_norm))
            all_true.append(normalizer.inverse_np(true_norm))

    pred = np.stack(all_pred, axis=0)
    true = np.stack(all_true, axis=0)

    pred_obs = pred[:, :, obs]
    true_obs = true[:, :, obs]

    metrics: Dict[str, float] = {}
    absolute_errors = np.linalg.norm(pred_obs - true_obs, axis=-1)

    true_scale = np.std(true_obs.reshape(-1, true_obs.shape[-1]), axis=0).mean() + 1e-6

    for horizon in horizons:
        horizon = min(int(horizon), max_horizon)
        error_slice = pred_obs[:, : horizon + 1] - true_obs[:, : horizon + 1]
        rmse = float(np.sqrt(np.mean(error_slice ** 2)))
        mae = float(np.mean(np.abs(error_slice)))
        nrmse = float(rmse / true_scale)
        metrics[f"rollout_rmse_h{horizon}"] = rmse
        metrics[f"rollout_mae_h{horizon}"] = mae
        metrics[f"rollout_nrmse_h{horizon}"] = nrmse

    divergence_steps: List[int] = []
    for sample_errors in absolute_errors:
        divergence_steps.append(_first_divergence_step(sample_errors, divergence_threshold))

    metrics["divergence_step_mean"] = float(np.mean(divergence_steps))
    metrics["divergence_step_std"] = float(np.std(divergence_steps))
    metrics["rollout_samples"] = float(sample_count)
    return metrics


def plot_rollout_samples(
    *,
    model: torch.nn.Module,
    trajectories_norm: np.ndarray,
    normalizer: StateNormalizer,
    t_eval: np.ndarray,
    state_labels: Sequence[str],
    observable_indices: Sequence[int],
    out_dir: str,
    num_samples: int,
    rng_seed: int,
    device: torch.device,
    prefix: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    rng = np.random.default_rng(rng_seed)
    sample_count = min(num_samples, trajectories_norm.shape[0])
    sampled_indices = rng.choice(trajectories_norm.shape[0], size=sample_count, replace=False)

    obs = list(observable_indices)
    horizon = trajectories_norm.shape[1] - 1

    with torch.no_grad():
        for rank, idx in enumerate(sampled_indices, start=1):
            traj_norm = trajectories_norm[idx]
            init = torch.tensor(traj_norm[0], dtype=torch.float32, device=device).unsqueeze(0)

            pred_norm = rollout_model(model, init, horizon).squeeze(0).cpu().numpy()
            true_norm = traj_norm

            pred = normalizer.inverse_np(pred_norm)
            true = normalizer.inverse_np(true_norm)

            rows = len(obs)
            fig, axes = plt.subplots(rows, 1, figsize=(10, 3.5 * rows), sharex=True)
            if rows == 1:
                axes = [axes]

            for axis_id, dim in enumerate(obs):
                ax = axes[axis_id]
                label = state_labels[dim] if dim < len(state_labels) else f"state_{dim}"
                ax.plot(t_eval, true[:, dim], "k-", label=f"True {label}")
                ax.plot(t_eval, pred[:, dim], "r--", label=f"Pred {label}")
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right")

            axes[-1].set_xlabel("Time")
            fig.suptitle(f"Autoregressive Rollout Sample {rank}")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{prefix}_rollout_{rank}.png"), dpi=150)
            plt.close(fig)
