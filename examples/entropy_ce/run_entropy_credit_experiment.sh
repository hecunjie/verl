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
METHOD_B_M_SAMPLES="${METHOD_B_M_SAMPLES:-4}"
METHOD_B_TOPK_ALT="${METHOD_B_TOPK_ALT:-10}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  examples/grpo_trainer/entropy_credit_experiment.py \
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
  --seed "${SEED}" \
  "$@"
