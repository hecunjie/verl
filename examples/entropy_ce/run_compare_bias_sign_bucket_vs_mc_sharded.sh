#!/usr/bin/env bash
set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
INPUT_DATA="${INPUT_DATA:-${HOME}/data/math_rl/dapo_math_17k_processed_train.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/entropy_check/compare_bias_sign_bucket_vs_mc}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
SEED="${SEED:-42}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-1.0}"
CANDIDATE_TOP_P="${CANDIDATE_TOP_P:-0.95}"
CANDIDATE_MAX_K="${CANDIDATE_MAX_K:-5}"
MAX_BRANCH_STEPS="${MAX_BRANCH_STEPS:-64}"

MC_M_SAMPLES_REF="${MC_M_SAMPLES_REF:-128}"
MC_TEMPERATURE="${MC_TEMPERATURE:-1.0}"
MC_TOP_P="${MC_TOP_P:-0.95}"
REAL_PATH_MODE="${REAL_PATH_MODE:-sampling}"
BIAS_METRICS_MODE="${BIAS_METRICS_MODE:-length_normalized}"
F_CONTINUATION_MODE="${F_CONTINUATION_MODE:-first_sentence}"
F_SENTENCE_MAX_NEW_TOKENS="${F_SENTENCE_MAX_NEW_TOKENS:-256}"
F_SENTENCE_STOP="${F_SENTENCE_STOP:-simple}"

VLLM_LOGPROBS_TOPK="${VLLM_LOGPROBS_TOPK:-20}"
VLLM_REQUEST_BATCH_CHUNK="${VLLM_REQUEST_BATCH_CHUNK:-64}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

BUCKET_GROUP_ROLLOUTS="${BUCKET_GROUP_ROLLOUTS:-16}"
BUCKET_NUM_BINS="${BUCKET_NUM_BINS:-100}"
BUCKET_MIN_POINTS_PER_BIN="${BUCKET_MIN_POINTS_PER_BIN:-4}"
BUCKET_PREFIX_KEY_MODE="${BUCKET_PREFIX_KEY_MODE:-sum}"
LOCAL_WINDOW_LEFT_TOKENS="${LOCAL_WINDOW_LEFT_TOKENS:-20}"
LOCAL_WINDOW_RIGHT_TOKENS="${LOCAL_WINDOW_RIGHT_TOKENS:-20}"
FBAR_MODE="${FBAR_MODE:-single_local}"

SAVE_TRACES="${SAVE_TRACES:-1}"
NO_PROGRESS="${NO_PROGRESS:-0}"
PROGRESS_ALL_RANKS="${PROGRESS_ALL_RANKS:-0}"
PROGRESS_ECHO="${PROGRESS_ECHO:-1}"

mkdir -p "${OUTPUT_DIR}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
unset HOST_IP 2>/dev/null || true

if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || [ "${NNODES}" -lt 1 ]; then
  echo "Invalid NNODES=${NNODES} (must be >=1)" >&2
  exit 1
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || [ "${NODE_RANK}" -lt 0 ] || [ "${NODE_RANK}" -ge "${NNODES}" ]; then
  echo "Invalid NODE_RANK=${NODE_RANK} (must satisfy 0 <= NODE_RANK < NNODES)" >&2
  exit 1
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [ "${NPROC_PER_NODE}" -lt 1 ]; then
  echo "Invalid NPROC_PER_NODE=${NPROC_PER_NODE} (must be >=1)" >&2
  exit 1
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
echo "[compare_bias_sign_bucket_vs_mc] NODE_RANK=${NODE_RANK}/${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} WORLD_SIZE=${WORLD_SIZE}" >&2

PIDS=()
for ((r = 0; r < NPROC_PER_NODE; r++)); do
  GLOBAL_RANK=$((NODE_RANK * NPROC_PER_NODE + r))
  EXTRA_ARGS=()
  if [ "${SAVE_TRACES}" != "1" ]; then
    EXTRA_ARGS+=(--no-save_traces)
  fi
  if [ "${NO_PROGRESS}" = "1" ]; then
    EXTRA_ARGS+=(--no_progress)
  fi
  if [ "${PROGRESS_ALL_RANKS}" = "1" ]; then
    EXTRA_ARGS+=(--progress_all_ranks)
  fi
  if [ "${PROGRESS_ECHO}" = "1" ]; then
    EXTRA_ARGS+=(--progress_echo)
  fi

  CUDA_VISIBLE_DEVICES="${r}" python3 examples/entropy_ce/compare_bias_sign_bucket_vs_mc.py \
    --input_data "${INPUT_DATA}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --seed "${SEED}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --entropy_threshold "${ENTROPY_THRESHOLD}" \
    --candidate_top_p "${CANDIDATE_TOP_P}" \
    --candidate_max_k "${CANDIDATE_MAX_K}" \
    --max_branch_steps "${MAX_BRANCH_STEPS}" \
    --mc_m_samples_ref "${MC_M_SAMPLES_REF}" \
    --mc_temperature "${MC_TEMPERATURE}" \
    --mc_top_p "${MC_TOP_P}" \
    --real_path_mode "${REAL_PATH_MODE}" \
    --bias_metrics_mode "${BIAS_METRICS_MODE}" \
    --f_continuation_mode "${F_CONTINUATION_MODE}" \
    --f_sentence_max_new_tokens "${F_SENTENCE_MAX_NEW_TOKENS}" \
    --f_sentence_stop "${F_SENTENCE_STOP}" \
    --vllm_logprobs_topk "${VLLM_LOGPROBS_TOPK}" \
    --vllm_request_batch_chunk "${VLLM_REQUEST_BATCH_CHUNK}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --bucket_group_rollouts "${BUCKET_GROUP_ROLLOUTS}" \
    --bucket_num_bins "${BUCKET_NUM_BINS}" \
    --bucket_min_points_per_bin "${BUCKET_MIN_POINTS_PER_BIN}" \
    --bucket_prefix_key_mode "${BUCKET_PREFIX_KEY_MODE}" \
    --local_window_left_tokens "${LOCAL_WINDOW_LEFT_TOKENS}" \
    --local_window_right_tokens "${LOCAL_WINDOW_RIGHT_TOKENS}" \
    --fbar_mode "${FBAR_MODE}" \
    --vllm_shard_rank "${GLOBAL_RANK}" \
    --vllm_shard_world_size "${WORLD_SIZE}" \
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

