#!/usr/bin/env bash
# 仅做验证集推理 + 打分（不训练）：利用 VERL 的 trainer.val_before_train + trainer.val_only。
# 每个 prompt 重复采样 VAL_N 次（actor_rollout_ref.rollout.val_kwargs.n），用于 mean@N / pass@N 分析。
#
# 指标说明（VERL 内置，无需后处理）：
# - mean@32：val-core/<data_source>/acc/mean@32（每题 32 次采样的平均正确率，再对题 macro 平均）
# - pass@32：val-core/<data_source>/acc/pass_strict@32（至少一次做对的比例）
# - 无偏 pass@32：val-core/.../acc/pass_unbiased@32（与 ``metric_utils.pass_at_k_unbiased`` 一致；当 n=k=32 时与 strict 同值）
#
# 用法示例（基座模型、多份 parquet 验证）::
#   VERL_ROOT=... MODEL_PATH=Qwen/Qwen3-4B \\
#   VAL_FILES_STR="['/data/math500_test.parquet','/data/amc23_test.parquet']" \\
#   bash examples/grpo_trainer/run_val_only_pass32_math.sh
#
# 从 RL checkpoint 评测（需 global_step_* 目录）::
#   RESUME_MODE=resume_path RESUME_FROM_PATH=/path/to/global_step_100 \\
#   bash examples/grpo_trainer/run_val_only_pass32_math.sh

set -euo pipefail

VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
PROJECT_NAME="${PROJECT_NAME:-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-val_only_pass32_math}"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/outputs/${EXPERIMENT_NAME}}"
mkdir -p "${OUTPUT_DIR}"

# 验证数据：Hydra 列表字符串，例如 "['/a/math500_test.parquet','/a/amc23_test.parquet']"
VAL_FILES_STR="${VAL_FILES_STR:-"['/path/to/math500_test.parquet']"}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

# 每个 prompt 的采样条数（mean@32 / pass@32 请设 32）
VAL_N="${VAL_N:-32}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"

ROLLOUT_LOGPROB_MB_PER_GPU="${ROLLOUT_LOGPROB_MB_PER_GPU:-16}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
REF_LOGPROB_MB_PER_GPU="${REF_LOGPROB_MB_PER_GPU:-4}"

# 基座评测：避免误从 OUTPUT_DIR 下 auto resume 旧 ckpt
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

# math_verify 验证（与训练脚本一致）
USE_MATH_VERIFY_VAL="${USE_MATH_VERIFY_VAL:-1}"
CUSTOM_REWARD_FUNCTION_PATH="${CUSTOM_REWARD_FUNCTION_PATH:-${VERL_ROOT}/examples/grpo_trainer/math_verify_val_reward.py}"

CUSTOM_REWARD_ARGS=()
if [ "${USE_MATH_VERIFY_VAL}" = "1" ]; then
  CUSTOM_REWARD_ARGS+=("custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH}")
  CUSTOM_REWARD_ARGS+=("custom_reward_function.name=compute_score")
fi

VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUTPUT_DIR}/validation_data}"
mkdir -p "${VALIDATION_DATA_DIR}"

RESUME_ARGS=()
RESUME_ARGS+=("trainer.resume_mode=${RESUME_MODE}")
if [ -n "${RESUME_FROM_PATH}" ]; then
  RESUME_ARGS+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  reward_model.enable=False \
  data.train_files="${VAL_FILES_STR}" \
  data.val_files="${VAL_FILES_STR}" \
  data.train_batch_size="${VAL_BATCH_SIZE}" \
  data.gen_batch_size="${VAL_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  "${RESUME_ARGS[@]}" \
  "${CUSTOM_REWARD_ARGS[@]}" \
  "$@"

echo ""
echo "=== 验证完成。原始生成见: ${VALIDATION_DATA_DIR}/<global_step>.jsonl（文件名与 checkpoint 的 global_steps 一致；未 resume 时多为 0.jsonl）"
echo "=== 计算严格 mean@${VAL_N} / pass@${VAL_N}（及无偏 pass@${VAL_N}）示例:"
echo "python3 ${VERL_ROOT}/examples/grpo_trainer/summarize_val_jsonl_pass_mean_at_k.py \\"
echo "  --jsonl ${VALIDATION_DATA_DIR}/<step>.jsonl --k ${VAL_N}"
