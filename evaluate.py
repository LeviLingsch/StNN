import argparse
import os
from typing import Dict

import numpy as np
import torch

from stnn_bench import (
    StateNormalizer,
    build_model,
    build_system,
    compute_rollout_metrics,
    flatten_transitions,
    load_config,
    load_or_generate_dataset,
    plot_rollout_samples,
    set_global_seed,
)
from stnn_bench.config_utils import dump_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Split to evaluate.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for evaluation artifacts.")
    return parser.parse_args()


def compute_one_step_mse(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    criterion = torch.nn.MSELoss(reduction="mean")
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch_x = x[start : start + batch_size].to(device)
            batch_y = y[start : start + batch_size].to(device)
            pred = model(batch_x)
            losses.append(criterion(pred, batch_y).item())
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("experiment", {}).get("seed", 42))
    set_global_seed(seed)

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
    run_dir = args.output_dir or os.path.dirname(args.checkpoint) or "output"
    os.makedirs(run_dir, exist_ok=True)

    data_bundle = load_or_generate_dataset(
        system=system,
        dataset_cfg=dataset_cfg,
        seed=seed,
        run_dir=run_dir,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    mean = checkpoint.get("normalizer_mean")
    std = checkpoint.get("normalizer_std")

    if mean is None or std is None:
        normalizer = StateNormalizer.from_trajectories(data_bundle.train)
    else:
        normalizer = StateNormalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))

    split = args.split
    trajectories = data_bundle.test if split == "test" else data_bundle.val
    trajectories_norm = normalizer.transform_np(trajectories)

    dt = float(dataset_cfg.get("dt", 0.1))
    model = build_model(config.get("model", {}), state_dim=system.state_dim, dt=dt).to(device)
    model.load_state_dict(checkpoint["model_state"])

    eval_cfg: Dict[str, object] = dict(config.get("evaluation", {}))
    horizons = [int(v) for v in eval_cfg.get("rollout_horizons", [20, 50, 100])]

    x_np, y_np = flatten_transitions(trajectories_norm)
    one_step_mse = compute_one_step_mse(
        model=model,
        x=torch.from_numpy(x_np),
        y=torch.from_numpy(y_np),
        device=device,
        batch_size=int(eval_cfg.get("eval_batch_size", 4096)),
    )

    rollout_metrics = compute_rollout_metrics(
        model=model,
        trajectories_norm=trajectories_norm,
        normalizer=normalizer,
        device=device,
        horizons=horizons,
        num_samples=int(eval_cfg.get("test_rollout_samples", 128)),
        observable_indices=system.observable_indices,
        divergence_threshold=float(eval_cfg.get("divergence_threshold", 1.5)),
        rng_seed=seed + 2026,
    )

    metrics = {
        f"{split}_one_step_mse": one_step_mse,
        **{f"{split}_{k}": v for k, v in rollout_metrics.items()},
    }
    dump_metrics(metrics, run_dir, filename=f"{split}_metrics_eval.json")

    if bool(eval_cfg.get("save_plots", True)):
        plot_rollout_samples(
            model=model,
            trajectories_norm=trajectories_norm,
            normalizer=normalizer,
            t_eval=data_bundle.t_eval,
            state_labels=system.state_labels,
            observable_indices=system.observable_indices,
            out_dir=os.path.join(run_dir, f"{split}_plots"),
            num_samples=int(eval_cfg.get("plot_samples", 5)),
            rng_seed=seed + 3030,
            device=device,
            prefix=split,
        )

    print(f"Evaluation complete on split '{split}'.")
    print(f"One-step MSE: {one_step_mse:.6f}")
    for horizon in horizons:
        key = f"{split}_rollout_rmse_h{horizon}"
        if key in metrics:
            print(f"RMSE@{horizon}: {metrics[key]:.6f}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
