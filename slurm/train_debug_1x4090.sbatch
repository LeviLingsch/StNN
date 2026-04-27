#!/bin/bash
#SBATCH --job-name=stnn-debug
#SBATCH --output=/cluster/home/llingsch/SAM/StNN/output/slurm/%x_%j.out
#SBATCH --error=/cluster/home/llingsch/SAM/StNN/output/slurm/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

set -euo pipefail

ENV_ACTIVATE="${STNN_PYTHON_ACTIVATE:-/cluster/home/llingsch/venvs/cameleon_venv/bin/activate}"
if [[ -f "${ENV_ACTIVATE}" ]]; then
	source "${ENV_ACTIVATE}"
else
	source /cluster/home/llingsch/SAM/.venv/bin/activate
fi
cd /cluster/home/llingsch/SAM/StNN

python train.py --config /cluster/home/llingsch/SAM/StNN/configs/pendulum_stnn.yaml --seed 123
