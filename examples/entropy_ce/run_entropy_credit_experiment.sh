#!/usr/bin/env bash
set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

# 单模型路径（按需改为本地模型目录或 HF 模型名）
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"

# 输入数据需符合 VERL RL 数据格式（含 prompt/data_source/reward_model.ground_truth）
INPUT_DATA="${INPUT_DATA:-${HOME}/data/math_rl/dapo_math_17k_processed_train.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/entropy_credit_outputs}"

# 8 卡多卡并行
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# 仅分析部分样本时可设置 >0；默认 -1 表示全量
MAX_SAMPLES="${MAX_SAMPLES:--1}"

# 每个 prompt 生成的 rollout 数（方案中默认 K=8）
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-8}"

# Phase2 默认使用方法 B（Outcome flip rate）
PHASE2_METHOD="${PHASE2_METHOD:-B}"
METHOD_B_M_SAMPLES="${METHOD_B_M_SAMPLES:-8}"
METHOD_B_TOPK_ALT="${METHOD_B_TOPK_ALT:-10}"
PHASE2_MAX_POSITIONS="${PHASE2_MAX_POSITIONS:-64}"
PHASE2_PROGRESS="${PHASE2_PROGRESS:-0}"
SAVE_CASE_TRACES="${SAVE_CASE_TRACES:-1}"
CONTEXT_WINDOW_TOKENS="${CONTEXT_WINDOW_TOKENS:-24}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"

# 推理后端：hf（默认）或 vllm（更快；熵特征为 top-K logprobs 近似）
BACKEND="${BACKEND:-hf}"
VLLM_LOGPROBS_TOPK="${VLLM_LOGPROBS_TOPK:-256}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

# 进度条：默认仅 rank0 显示；每卡各一条可设 PROGRESS_ALL_RANKS=1 或 export TQDM_DISABLE=1 关闭
PROGRESS_ALL_RANKS="${PROGRESS_ALL_RANKS:-0}"
EXTRA_ARGS=()
if [ "${PROGRESS_ALL_RANKS}" = "1" ]; then
  EXTRA_ARGS+=(--progress_all_ranks)
fi
if [ "${PHASE2_PROGRESS}" = "1" ]; then
  EXTRA_ARGS+=(--phase2_progress)
fi
if [ "${SAVE_CASE_TRACES}" = "1" ]; then
  EXTRA_ARGS+=(--save_case_traces)
fi

mkdir -p "${OUTPUT_DIR}"


torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  examples/entropy_ce/entropy_credit_experiment.py \
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
  --backend "${BACKEND}" \
  --vllm_logprobs_topk "${VLLM_LOGPROBS_TOPK}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
