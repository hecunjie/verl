#!/usr/bin/env bash
# 推荐：vLLM + 多卡时不要用 torchrun。本脚本为「每 GPU 一个独立 python 进程」，避免 MASTER_ADDR/TCPStore 与 vLLM 子进程冲突。
#
# 用法（在 VERL 根目录）:
#   bash examples/entropy_ce/run_entropy_credit_experiment_vllm_sharded.sh
# 或:
#   NPROC_PER_NODE=4 MODEL_PATH=/path/to/model bash examples/entropy_ce/run_entropy_credit_experiment_vllm_sharded.sh
#
# 每个子进程需设置:
#   --vllm_shard_rank R --vllm_shard_world_size N
# 且本进程只看见一张卡: CUDA_VISIBLE_DEVICES=R
#
set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
INPUT_DATA="${INPUT_DATA:-${HOME}/data/math_rl/dapo_math_17k_processed_train.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/entropy_credit_outputs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-8}"
SEED="${SEED:-42}"
VLLM_LOGPROBS_TOPK="${VLLM_LOGPROBS_TOPK:-256}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
unset HOST_IP 2>/dev/null || true
# 仍可与 legacy engine 同用: export VLLM_USE_V1=0

PHASE2_METHOD="${PHASE2_METHOD:-B}"
METHOD_B_M_SAMPLES="${METHOD_B_M_SAMPLES:-8}"
METHOD_B_TOPK_ALT="${METHOD_B_TOPK_ALT:-10}"
PHASE2_MAX_POSITIONS="${PHASE2_MAX_POSITIONS:-64}"
CONTEXT_WINDOW_TOKENS="${CONTEXT_WINDOW_TOKENS:-24}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [ "${SAVE_CASE_TRACES:-1}" = "1" ]; then
  EXTRA_ARGS+=(--save_case_traces)
fi
if [ "${VLLM_LEGACY_ENGINE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--vllm_legacy_engine)
fi

PIDS=()
for ((r = 0; r < NPROC_PER_NODE; r++)); do
  CUDA_VISIBLE_DEVICES="${r}" python3 examples/entropy_ce/entropy_credit_experiment.py \
    --input_data "${INPUT_DATA}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --rollouts_per_prompt "${ROLLOUTS_PER_PROMPT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --phase2_method "${PHASE2_METHOD}" \
    --method_b_m_samples "${METHOD_B_M_SAMPLES}" \
    --method_b_topk_alt "${METHOD_B_TOPK_ALT}" \
    --phase2_max_positions "${PHASE2_MAX_POSITIONS}" \
    --context_window_tokens "${CONTEXT_WINDOW_TOKENS}" \
    --seed "${SEED}" \
    --backend vllm \
    --vllm_shard_rank "${r}" \
    --vllm_shard_world_size "${NPROC_PER_NODE}" \
    --vllm_logprobs_topk "${VLLM_LOGPROBS_TOPK}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    "${EXTRA_ARGS[@]}" \
    "$@" &
  PIDS+=("$!")
done

EXIT=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    EXIT=1
  fi
done
exit "${EXIT}"
