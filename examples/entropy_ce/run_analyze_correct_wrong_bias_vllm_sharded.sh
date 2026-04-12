#!/usr/bin/env bash
# vLLM + 多卡：每 GPU 一个独立 python 进程。与 run_calibrate_mc_variance_vllm_sharded.sh 用法一致。
#
# 单机（默认）：NNODES=1, NODE_RANK=0，world_size=NPROC_PER_NODE，rank=0..NPROC_PER_NODE-1。
#
# 多机：各节点挂载同一 OUTPUT_DIR（NFS/共享盘），在每台上用相同 MAX_SAMPLES/SEED/INPUT_DATA 等启动；
#   NNODES=节点总数，NODE_RANK=本机编号（0 起，每台不同）。
#   全局 rank = NODE_RANK * NPROC_PER_NODE + 本机 GPU 下标，world_size = NNODES * NPROC_PER_NODE。
#   合并 jsonl / summary 仅「全局 rank 0」进程执行（第一台机器上的 GPU0）。
# 多机时建议显式设置本机可被访问的 IP（部分 vLLM 版本会用到）:
#   export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
#
# 进度条：
#   默认 rank0 外层 tqdm + 嵌套 tqdm（rollout / 高熵位置 / F 估计）；PROGRESS_NESTED=0 可关嵌套（只保留按 prompt 一条杠）
#   PROGRESS_ALL_RANKS=1：8 卡各一条 tqdm（屏会很乱，一般不推荐）
#   PROGRESS_ECHO=1：rank0 每进入一条 prompt 在 stderr 打一行累计秒数（便于估总时长）
# 关闭：--no_progress 或 export TQDM_DISABLE=1
#
# 全量 rollout 审计（题干 + 每条完整回答，含对错）：默认开启，写入 OUTPUT_DIR/rollouts_archive_merged.jsonl；
#   关闭：SAVE_ROLLOUTS_ARCHIVE=0
#
# bias/F 量纲（方案 1）：BIAS_METRICS_MODE=length_normalized 时，MC 续写为 (熵和)/(续写步数) 再平均，与 entropy_t 同量纲；
#   默认 raw=续写熵之和。
#
# 用法（在 VERL 根目录）:
#   bash examples/entropy_ce/run_analyze_correct_wrong_bias_vllm_sharded.sh
#
# 多机示例（2 台 × 8 卡，共享 OUTPUT_DIR）:
#   节点0: NNODES=2 NODE_RANK=0 OUTPUT_DIR=/nfs/exp1 ... bash examples/entropy_ce/run_analyze_correct_wrong_bias_vllm_sharded.sh
#   节点1: NNODES=2 NODE_RANK=1 OUTPUT_DIR=/nfs/exp1 ... bash examples/entropy_ce/run_analyze_correct_wrong_bias_vllm_sharded.sh
#
set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
INPUT_DATA="${INPUT_DATA:-${HOME}/data/math_rl/dapo_math_17k_processed_train.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/entropy_check/pair_bias}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-8}"
SEED="${SEED:-42}"
MC_M_SAMPLES="${MC_M_SAMPLES:-64}"
# F 估计：默认 beam（与 analyze 脚本默认一致）；改 MC 时设 F_ESTIMATOR=mc
F_ESTIMATOR="${F_ESTIMATOR:-beam}"
F_BEAM_WIDTH="${F_BEAM_WIDTH:-10}"
# F 续写：full=一次续写到 MAX_NEW_TOKENS；first_sentence=单次最多 F_SENTENCE_MAX_NEW_TOKENS 再截断到首句（可批量）
F_CONTINUATION_MODE="${F_CONTINUATION_MODE:-full}"
F_SENTENCE_MAX_NEW_TOKENS="${F_SENTENCE_MAX_NEW_TOKENS:-256}"
# 句末判定：simple | pysbd（需 pip install pysbd）
F_SENTENCE_STOP="${F_SENTENCE_STOP:-simple}"
# bar F_t 候选：默认 topp（高熵步自动多取几个，上限 candidate_max_k）；legacy 固定 k 见 CANDIDATE_MODE=fixed
CANDIDATE_MODE="${CANDIDATE_MODE:-topp}"
CANDIDATE_TOP_P="${CANDIDATE_TOP_P:-0.9}"
CANDIDATE_MAX_K="${CANDIDATE_MAX_K:-5}"
TOPK_ALT="${TOPK_ALT:-3}"
TOP_ENTROPY_RATIO="${TOP_ENTROPY_RATIO:-0.10}"
# 高熵位置数 = min(本值, ceil(TOP_ENTROPY_RATIO * 序列长度))
MAX_POSITIONS_PER_ROLLOUT="${MAX_POSITIONS_PER_ROLLOUT:-500}"
# 与 examples/grpo_trainer/run_grpo_r1_distill_7b_dapo_data.sh 中 data.max_response_length 对齐；
# R1/Distill 长 CoT 若仍用 256，常在未写出 \\boxed{} 前截断 → 8 条 rollout 同判错 → n_skip_no_mixed 饱和。
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
VLLM_LOGPROBS_TOPK="${VLLM_LOGPROBS_TOPK:-20}"
# 单次 vLLM 请求批大小（rollout 与 F 估计分块）；OOM 时改为 8 或 16
VLLM_REQUEST_BATCH_CHUNK="${VLLM_REQUEST_BATCH_CHUNK:-64}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
PROGRESS_ALL_RANKS="${PROGRESS_ALL_RANKS:-0}"
PROGRESS_NESTED="${PROGRESS_NESTED:-1}"
PROGRESS_ECHO="${PROGRESS_ECHO:-0}"
CONTEXT_WINDOW_TOKENS="${CONTEXT_WINDOW_TOKENS:-64}"
PER_SAMPLE_JSONL_SUBDIR="${PER_SAMPLE_JSONL_SUBDIR:-per_sample}"
NO_PER_SAMPLE_JSONL="${NO_PER_SAMPLE_JSONL:-0}"
SAVE_ROLLOUTS_ARCHIVE="${SAVE_ROLLOUTS_ARCHIVE:-1}"
BIAS_METRICS_MODE="${BIAS_METRICS_MODE:-raw}"

export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
unset HOST_IP 2>/dev/null || true

if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || [ "${NNODES}" -lt 1 ]; then
  echo "Invalid NNODES=${NNODES} (must be a positive integer)" >&2
  exit 1
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || [ "${NODE_RANK}" -ge "${NNODES}" ]; then
  echo "Invalid NODE_RANK=${NODE_RANK} (must satisfy 0 <= NODE_RANK < NNODES, NNODES=${NNODES})" >&2
  exit 1
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [ "${NPROC_PER_NODE}" -lt 1 ]; then
  echo "Invalid NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
echo "[run_analyze_correct_wrong_bias] NODE_RANK=${NODE_RANK}/${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} WORLD_SIZE=${WORLD_SIZE}" >&2

mkdir -p "${OUTPUT_DIR}"
# 如需 legacy vLLM engine：在运行前 export VLLM_USE_V1=0

PIDS=()
for ((r = 0; r < NPROC_PER_NODE; r++)); do
  EXTRA_ARGS=()
  if [ "${PROGRESS_ALL_RANKS}" = "1" ]; then
    EXTRA_ARGS+=(--progress_all_ranks)
  fi
  if [ "${PROGRESS_NESTED}" != "1" ]; then
    EXTRA_ARGS+=(--no-progress_nested)
  fi
  if [ "${PROGRESS_ECHO}" = "1" ]; then
    EXTRA_ARGS+=(--progress_echo)
  fi
  if [ "${NO_PER_SAMPLE_JSONL}" = "1" ]; then
    EXTRA_ARGS+=(--no_per_sample_jsonl)
  fi
  if [ "${SAVE_ROLLOUTS_ARCHIVE}" != "1" ]; then
    EXTRA_ARGS+=(--no-save-rollouts-archive)
  fi
  EXTRA_ARGS+=(--bias_metrics_mode "${BIAS_METRICS_MODE}")
  EXTRA_ARGS+=(--f_continuation_mode "${F_CONTINUATION_MODE}")
  EXTRA_ARGS+=(--f_sentence_max_new_tokens "${F_SENTENCE_MAX_NEW_TOKENS}")
  EXTRA_ARGS+=(--f_sentence_stop "${F_SENTENCE_STOP}")
  GLOBAL_RANK=$((NODE_RANK * NPROC_PER_NODE + r))
  CUDA_VISIBLE_DEVICES="${r}" python3 examples/entropy_ce/analyze_correct_wrong_bias.py \
    --input_data "${INPUT_DATA}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --rollouts_per_prompt "${ROLLOUTS_PER_PROMPT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --f_estimator "${F_ESTIMATOR}" \
    --f_beam_width "${F_BEAM_WIDTH}" \
    --mc_m_samples "${MC_M_SAMPLES}" \
    --candidate_mode "${CANDIDATE_MODE}" \
    --candidate_top_p "${CANDIDATE_TOP_P}" \
    --candidate_max_k "${CANDIDATE_MAX_K}" \
    --topk_alt "${TOPK_ALT}" \
    --top_entropy_ratio "${TOP_ENTROPY_RATIO}" \
    --max_positions_per_rollout "${MAX_POSITIONS_PER_ROLLOUT}" \
    --context_window_tokens "${CONTEXT_WINDOW_TOKENS}" \
    --per_sample_jsonl_subdir "${PER_SAMPLE_JSONL_SUBDIR}" \
    --seed "${SEED}" \
    --vllm_logprobs_topk "${VLLM_LOGPROBS_TOPK}" \
    --vllm_request_batch_chunk "${VLLM_REQUEST_BATCH_CHUNK}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
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
