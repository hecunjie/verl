#!/usr/bin/env bash
# Submit GRPO DAPO math training to an existing Ray cluster.
#
# Usage (direct run, no edits needed):
#   bash examples/grpo_trainer/submit_grpo_r1_distill_7b_dapo_ray.sh
#
# Optional overrides (examples):
#   RAY_ADDRESS="http://10.144.24.240:8265" bash examples/grpo_trainer/submit_grpo_r1_distill_7b_dapo_ray.sh
#   WANDB_API_KEY="***" WANDB_MODE=online bash examples/grpo_trainer/submit_grpo_r1_distill_7b_dapo_ray.sh
#   bash examples/grpo_trainer/submit_grpo_r1_distill_7b_dapo_ray.sh trainer.nnodes=2 trainer.n_gpus_per_node=8
#
# Notes:
# - This script intentionally does NOT hardcode WANDB_API_KEY.
# - Ray job submission may record the entrypoint; avoid pasting secrets into commands.

set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

# Ray Job submission server address (dashboard endpoint).
# Default points to the address you previously used in this workspace.
RAY_ADDRESS="${RAY_ADDRESS:-${RAY_JOB_ADDRESS:-http://10.144.24.240:8265}}"

# ---- vLLM / tokenizer envs (cluster-wide) ----
VLLM_USE_V1="${VLLM_USE_V1:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ---- WandB envs (cluster-wide) ----
# Strongly recommended: provide WANDB_API_KEY via secret injection rather than typing it in terminal.
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-verl}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}" # online | offline | disabled
WANDB_DIR="${WANDB_DIR:-${HOME}/wandb}"

# Make it runnable out-of-the-box: if user didn't provide WANDB_API_KEY, don't try online logging.
if [[ -z "${WANDB_API_KEY}" && "${WANDB_MODE}" == "online" ]]; then
  WANDB_MODE="offline"
fi

# Build runtime_env JSON safely. Avoid printing secrets.
runtime_env_json="$(
  python3 - <<'PY'
import json
import os

env_vars = {
  "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
  "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
  "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
  "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "verl"),
  "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
  "WANDB_DIR": os.environ.get("WANDB_DIR", ""),
}

# Optional (only include if explicitly provided)
wandb_entity = os.environ.get("WANDB_ENTITY", "")
if wandb_entity:
  env_vars["WANDB_ENTITY"] = wandb_entity

wandb_key = os.environ.get("WANDB_API_KEY", "")
if wandb_key:
  env_vars["WANDB_API_KEY"] = wandb_key

runtime_env = {
  "working_dir": os.getcwd(),
  "env_vars": env_vars,
}
print(json.dumps(runtime_env, ensure_ascii=False))
PY
)"

echo "Submitting Ray job to: ${RAY_ADDRESS}"
echo "Working dir (runtime_env.working_dir): ${VERL_ROOT}"
echo "Env vars: VLLM_USE_V1=${VLLM_USE_V1}, WANDB_PROJECT=${WANDB_PROJECT}, WANDB_MODE=${WANDB_MODE}"
if [[ -n "${WANDB_ENTITY}" ]]; then
  echo "Env vars: WANDB_ENTITY=${WANDB_ENTITY}"
fi
if [[ "${WANDB_MODE}" != "disabled" && -z "${WANDB_API_KEY}" ]]; then
  echo "Note: WANDB_API_KEY is not set, using WANDB_MODE=${WANDB_MODE}." >&2
fi

# Pass through any Hydra overrides from CLI to the underlying training script.
# Example overrides:
#   trainer.nnodes=2 trainer.n_gpus_per_node=8 trainer.experiment_name=...
ray job submit --address="${RAY_ADDRESS}" \
  --runtime-env-json="${runtime_env_json}" \
  -- \
  bash examples/grpo_trainer/run_grpo_r1_distill_7b_dapo_data.sh \
  trainer.nnodes=2 \
  trainer.n_gpus_per_node=8 \
  trainer.experiment_name=dapo_train_math500_aime24_val_2node16gpu \
  "$@"

