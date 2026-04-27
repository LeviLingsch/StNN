#!/bin/bash
#SBATCH --job-name=stnn-rebuttal
#SBATCH --output=/cluster/home/llingsch/SAM/StNN/output/slurm/%x_%j.out
#SBATCH --error=/cluster/home/llingsch/SAM/StNN/output/slurm/%x_%j.err
#SBATCH --time=02:30:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/train_1x4090.sbatch <config_path> [extra train.py args]"
  exit 1
fi

CONFIG_PATH="$1"
shift

module load stack/2024-06 python_cuda nccl eth_proxy
source /cluster/home/llingsch/SAM/Phaedra/scripts/setup_env.sh

cd /cluster/home/llingsch/SAM/StNN
mkdir -p /cluster/home/llingsch/SAM/StNN/output/slurm
mkdir -p /cluster/home/llingsch/SAM/StNN/output/rebuttal_runs
mkdir -p /cluster/home/llingsch/SAM/StNN/output/datasets

export WANDB__SERVICE_WAIT=300
export WANDB_SILENT=true

python train.py --config "$CONFIG_PATH" "$@"
