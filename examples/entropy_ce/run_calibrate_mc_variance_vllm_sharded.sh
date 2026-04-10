#!/usr/bin/env bash
# vLLM + 多卡：每 GPU 一个独立 python 进程（勿用 torchrun），与 run_entropy_credit_experiment_vllm_sharded.sh 一致。
#
# 默认仅 rank0 显示 tqdm；PROGRESS_ALL_RANKS=1 显示每卡一条
# 嵌套进度（rollout batches→prefixes→MC batched）：PROGRESS_NESTED=1（推荐只看 rank0，否则终端很乱）
# 关闭：传参 --no_progress 或 export TQDM_DISABLE=1
# 单次 llm.generate 最大并发序列数（rollout 与 MC）：VLLM_REQUEST_BATCH_CHUNK（默认 64，与 analyze_correct_wrong_bias 一致）
# 高熵前缀数：k = min(MAX_POSITIONS_PER_ROLLOUT, ceil(TOP_ENTROPY_RATIO * 生成长度))，默认 0.10 与 100 → 即 min(10%*T, 100)
#
# 用法（在 VERL 根目录）:
#   bash examples/entropy_ce/run_calibrate_mc_variance_vllm_sharded.sh
# 或:
#   NPROC_PER_NODE=4 MODEL_PATH=/path/to/model INPUT_DATA=/path/to/data.parquet bash examples/entropy_ce/run_calibrate_mc_variance_vllm_sharded.sh
#
set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
INPUT_DATA="${INPUT_DATA:-${HOME}/data/math_rl/dapo_math_17k_processed_train.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/entropy_check/calibrate_mc}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-8}"
SEED="${SEED:-42}"
M_GRID="${M_GRID:-2,4,8,12,16}"
TOP_ENTROPY_RATIO="${TOP_ENTROPY_RATIO:-0.10}"
MAX_POSITIONS_PER_ROLLOUT="${MAX_POSITIONS_PER_ROLLOUT:-100}"
CI95_THRESHOLD="${CI95_THRESHOLD:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
VLLM_LOGPROBS_TOPK="${VLLM_LOGPROBS_TOPK:-20}"
VLLM_REQUEST_BATCH_CHUNK="${VLLM_REQUEST_BATCH_CHUNK:-64}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
PROGRESS_ALL_RANKS="${PROGRESS_ALL_RANKS:-0}"
PROGRESS_NESTED="${PROGRESS_NESTED:-0}"

export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
unset HOST_IP 2>/dev/null || true

mkdir -p "${OUTPUT_DIR}"
# 如需 legacy vLLM engine：在运行前 export VLLM_USE_V1=0

PIDS=()
for ((r = 0; r < NPROC_PER_NODE; r++)); do
  EXTRA_ARGS=()
  if [ "${PROGRESS_ALL_RANKS}" = "1" ]; then
    EXTRA_ARGS+=(--progress_all_ranks)
  fi
  if [ "${PROGRESS_NESTED}" = "1" ]; then
    EXTRA_ARGS+=(--progress_nested)
  fi
  CUDA_VISIBLE_DEVICES="${r}" python3 examples/entropy_ce/calibrate_mc_variance.py \
    --input_data "${INPUT_DATA}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --rollouts_per_prompt "${ROLLOUTS_PER_PROMPT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --m_grid "${M_GRID}" \
    --top_entropy_ratio "${TOP_ENTROPY_RATIO}" \
    --max_positions_per_rollout "${MAX_POSITIONS_PER_ROLLOUT}" \
    --ci95_threshold "${CI95_THRESHOLD}" \
    --seed "${SEED}" \
    --vllm_logprobs_topk "${VLLM_LOGPROBS_TOPK}" \
    --vllm_request_batch_chunk "${VLLM_REQUEST_BATCH_CHUNK}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_shard_rank "${r}" \
    --vllm_shard_world_size "${NPROC_PER_NODE}" \
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
