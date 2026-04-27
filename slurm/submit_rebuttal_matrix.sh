#!/bin/bash
set -euo pipefail

ROOT_DIR="/cluster/home/llingsch/SAM/StNN"
CONFIG_DIR="${ROOT_DIR}/configs"
SBATCH_SCRIPT="${ROOT_DIR}/slurm/train_1x4090.sbatch"
LOG_DIR="${ROOT_DIR}/output/slurm"

mkdir -p "${LOG_DIR}"

SEED_OVERRIDE="${1:-}"
if [[ -n "${SEED_OVERRIDE}" ]]; then
  EXTRA_ARGS=(--seed "${SEED_OVERRIDE}")
else
  EXTRA_ARGS=()
fi

CONFIGS=(
  "pendulum_stnn.yaml"
  # "pendulum_mlp.yaml"
  # "pendulum_neural_ode.yaml"
  "duffing_stnn.yaml"
  # "duffing_mlp.yaml"
  # "duffing_neural_ode.yaml"
  "duffing_neural_ode_stnn_field.yaml"
  "forced_duffing_stnn.yaml"
  # "forced_duffing_mlp.yaml"
  # "forced_duffing_neural_ode.yaml"
  "forced_duffing_neural_ode_stnn_field.yaml"
)

MANIFEST="${LOG_DIR}/submission_manifest_$(date +%Y%m%d_%H%M%S).csv"
echo "job_id,config,seed_override" > "${MANIFEST}"

for config_file in "${CONFIGS[@]}"; do
  config_path="${CONFIG_DIR}/${config_file}"
  job_id=$(sbatch --parsable "${SBATCH_SCRIPT}" "${config_path}" "${EXTRA_ARGS[@]}")
  echo "${job_id},${config_file},${SEED_OVERRIDE:-config_default}" >> "${MANIFEST}"
  echo "submitted ${config_file} as job ${job_id}"
done

echo "submission manifest: ${MANIFEST}"
