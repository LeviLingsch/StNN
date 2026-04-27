# StNN Non-Polynomial Benchmark Suite

This folder contains a config-driven benchmark framework for:

- Improved StNN training on damped pendulum
- Two additional reviewer-requested systems: unforced Duffing and forced Duffing
- Two baselines in the same setting: MLP and Neural ODE
- NeuralODE vector-field comparison: standard MLP field vs StNN-structured field
- Unified one-step and long-horizon rollout evaluation
- 1x RTX 4090 Slurm launch scripts

## Main Entry Points

- `train.py`: Train and evaluate a single config
- `evaluate.py`: Evaluate a saved checkpoint
- `configs/*.yaml`: Experiment configs (base 3 systems x 3 models plus field-comparison runs)
- `slurm/train_1x4090.sbatch`: Single-run Slurm launcher
- `slurm/submit_rebuttal_matrix.sh`: Submit all 11 configs

## Quick Start

Run one experiment locally:

```bash
cd /cluster/home/llingsch/SAM/StNN
python train.py --config configs/pendulum_stnn.yaml
```

Evaluate an existing checkpoint:

```bash
python evaluate.py \
  --config configs/pendulum_stnn.yaml \
  --checkpoint output/rebuttal_runs/pendulum_stnn/checkpoint_best.pt
```

## Slurm Usage (1x RTX 4090)

Submit one experiment:

```bash
cd /cluster/home/llingsch/SAM/StNN
sbatch slurm/train_1x4090.sbatch /cluster/home/llingsch/SAM/StNN/configs/pendulum_stnn.yaml
```

Optional: override the Python environment used by the Slurm launcher:

```bash
export STNN_PYTHON_ACTIVATE=/path/to/venv/bin/activate
sbatch slurm/train_1x4090.sbatch /cluster/home/llingsch/SAM/StNN/configs/pendulum_stnn.yaml
```

Submit full matrix (optionally override seed):

```bash
bash /cluster/home/llingsch/SAM/StNN/slurm/submit_rebuttal_matrix.sh
# optional seed override
bash /cluster/home/llingsch/SAM/StNN/slurm/submit_rebuttal_matrix.sh 123
```

Each run writes:

- `config_snapshot.yaml`
- `history.jsonl`
- `checkpoint_best.pt`
- `checkpoint_last.pt`
- `test_metrics.json`
- rollout plots in `plots/`

## StNN Improvements Implemented

- Residual update dynamics in the StNN forward pass
- Optional input/output LayerNorm controls (disabled by default for low-dimensional oscillators)
- Multi-step rollout-aware loss with scheduled teacher forcing
- Gradient clipping and warmup-cosine schedule
- Orthogonality regularization for structured matrix stability

## NeuralODE Field Comparison

- MLP field baseline: `configs/duffing_neural_ode.yaml`, `configs/forced_duffing_neural_ode.yaml`
- StNN field variant: `configs/duffing_neural_ode_stnn_field.yaml`, `configs/forced_duffing_neural_ode_stnn_field.yaml`

## Fairness Controls Across Models

- Same train/val/test generation protocol per system
- Same rollout loss and metric pipeline
- Same horizon metrics and divergence metric
- Same output artifact format for direct comparisons
# StNN
