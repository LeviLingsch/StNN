from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .systems import DynamicalSystem


@dataclass
class DatasetBundle:
    t_eval: np.ndarray
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class StateNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.clip(std.astype(np.float32), a_min=1e-6, a_max=None)

    @classmethod
    def from_trajectories(cls, trajectories: np.ndarray) -> "StateNormalizer":
        flat = trajectories.reshape(-1, trajectories.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        return cls(mean=mean, std=std)

    def transform_np(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_np(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def transform_torch(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return (values - mean) / std

    def inverse_torch(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return values * std + mean


class TrajectoryTensorDataset(Dataset):
    def __init__(self, trajectories: torch.Tensor) -> None:
        self.trajectories = trajectories

    def __len__(self) -> int:
        return self.trajectories.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.trajectories[index]


def flatten_transitions(trajectories: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = trajectories[:, :-1, :].reshape(-1, trajectories.shape[-1])
    y = trajectories[:, 1:, :].reshape(-1, trajectories.shape[-1])
    return x.astype(np.float32), y.astype(np.float32)


def _save_dataset_npz(path: str, bundle: DatasetBundle) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        t_eval=bundle.t_eval,
        train=bundle.train,
        val=bundle.val,
        test=bundle.test,
    )


def _load_dataset_npz(path: str) -> DatasetBundle:
    loaded = np.load(path)
    return DatasetBundle(
        t_eval=loaded["t_eval"].astype(np.float32),
        train=loaded["train"].astype(np.float32),
        val=loaded["val"].astype(np.float32),
        test=loaded["test"].astype(np.float32),
    )


def load_or_generate_dataset(
    *,
    system: DynamicalSystem,
    dataset_cfg: Dict[str, object],
    seed: int,
    run_dir: str,
) -> DatasetBundle:
    t_span = tuple(float(v) for v in dataset_cfg.get("t_span", (0.0, 10.0)))
    dt = float(dataset_cfg.get("dt", 0.1))

    train_n = int(dataset_cfg.get("train_trajectories", 1000))
    val_n = int(dataset_cfg.get("val_trajectories", 200))
    test_n = int(dataset_cfg.get("test_trajectories", 200))

    cache_path = dataset_cfg.get("cache_path")
    if cache_path:
        cache_path = str(cache_path)
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(run_dir, cache_path)

        if os.path.exists(cache_path):
            return _load_dataset_npz(cache_path)

    train = system.generate_trajectories(
        num_trajectories=train_n, t_span=t_span, dt=dt, seed=seed + 11
    )
    val = system.generate_trajectories(
        num_trajectories=val_n, t_span=t_span, dt=dt, seed=seed + 29
    )
    test = system.generate_trajectories(
        num_trajectories=test_n, t_span=t_span, dt=dt, seed=seed + 47
    )

    bundle = DatasetBundle(
        t_eval=train.t_eval,
        train=train.trajectories,
        val=val.trajectories,
        test=test.trajectories,
    )

    if cache_path:
        _save_dataset_npz(cache_path, bundle)

    return bundle
