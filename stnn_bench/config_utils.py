import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config at {config_path}. Expected a mapping.")
    return config


def resolve_run_directory(config: Dict[str, Any]) -> Tuple[str, str]:
    experiment_cfg = config.get("experiment", {})
    output_dir = experiment_cfg.get("output_dir", "output")
    run_name = experiment_cfg.get("run_name")

    if not run_name:
        system_name = config.get("system", {}).get("name", "system")
        model_name = config.get("model", {}).get("type", "model")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{system_name}_{model_name}_{timestamp}"

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name


def save_config_snapshot(config: Dict[str, Any], run_dir: str) -> None:
    path = os.path.join(run_dir, "config_snapshot.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_metrics(metrics: Dict[str, Any], run_dir: str, filename: str = "metrics.json") -> None:
    path = os.path.join(run_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
