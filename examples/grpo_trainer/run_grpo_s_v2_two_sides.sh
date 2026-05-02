#!/usr/bin/env bash
# GRPO-S（arXiv:2508.04349，Sequence-Level GRPO）启动脚本
# - 与 run_fepo_v2_lowtail_two_sides.sh 相同的数据路径 / batch / rollout 习惯，便于对照
# - adv_estimator=grpo_s：序列级熵 shaping + 组内 GRPO 标准化
#
# 论文 §3.1 Implementation Details（主表/主实验采用的一组「最好」设置）：
#   - global batch size 128，group size（rollout.n）16，lr 1e-6
#   - 生成：temperature=1.0，top-p=1.0；max prompt 2048，max response 4096
#   - shaping：β1=1，β2=0.1；熵 clip ε_low=0.2，ε_high=0.28
#   （敏感性分析显示 β2 从 0.1 提到 0.2 会损害表现，故默认保持 0.1。）

set -euo pipefail

VERL_ROOT="${VERL_ROOT:-/Users/hecunjie/Documents/python_lianxi/github_source/verl}"
cd "${VERL_ROOT}"

# -----------------------------
# Core experiment identity
# -----------------------------
MODEL_PATH="${MODEL_PATH:-/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B}"
PROJECT_NAME="${PROJECT_NAME:-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_s_v2_two_sides_qwen3_4b}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/qwen3-4b-grpo-s-v2}"
mkdir -p "${OUTPUT_DIR}"

# -----------------------------
# Data
# -----------------------------
TRAIN_FILES="${TRAIN_FILES:-/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet}"
MATH500_VAL="${MATH500_VAL:-/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/math500_test.parquet}"
AIME24_VAL="${AIME24_VAL:-/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet}"
USE_MATH_VERIFY_VAL="${USE_MATH_VERIFY_VAL:-1}"
CUSTOM_REWARD_FUNCTION_PATH="${CUSTOM_REWARD_FUNCTION_PATH:-${VERL_ROOT}/examples/grpo_trainer/math_verify_val_reward.py}"

CUSTOM_REWARD_ARGS=()
if [ "${USE_MATH_VERIFY_VAL}" = "1" ]; then
  CUSTOM_REWARD_ARGS+=("custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH}")
  CUSTOM_REWARD_ARGS+=("custom_reward_function.name=compute_score")
fi

# -----------------------------
# Training scale
# -----------------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"

# -----------------------------
# PPO / rollout
# -----------------------------
ACTOR_LR="${ACTOR_LR:-1e-6}"
WARM_UP_RATIO="${WARM_UP_RATIO:-0.05}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ROLLOUT_LOGPROB_MB_PER_GPU="${ROLLOUT_LOGPROB_MB_PER_GPU:-16}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
REF_LOGPROB_MB_PER_GPU="${REF_LOGPROB_MB_PER_GPU:-4}"

# -----------------------------
# Validation generation
# -----------------------------
VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"

# -----------------------------
# GRPO-S（论文 β1/β2 与熵 clip；与 GTPO 主实验一致）
# - 默认 dapo + geometric：对齐论文附录 C（失败样本 r=-1 + 几何负项）
# - 若要回退到 VERL 常见 0/1 标量奖励语义，可设：
#   GRPOS_OUTCOME_CONVENTION=grpo  GRPOS_NEGATIVE_ENTROPY_NORM=arithmetic
# - 若训练不稳定，建议先关闭负样本熵 shaping：
#   GRPOS_SHAPE_NEGATIVE=false
# -----------------------------
GRPOS_BETA1="${GRPOS_BETA1:-1.0}"
GRPOS_BETA2="${GRPOS_BETA2:-0.1}"
GRPOS_ENTROPY_CLIP_LOW="${GRPOS_ENTROPY_CLIP_LOW:-0.2}"
GRPOS_ENTROPY_CLIP_HIGH="${GRPOS_ENTROPY_CLIP_HIGH:-0.28}"
GRPOS_SUCCESS_THRESHOLD="${GRPOS_SUCCESS_THRESHOLD:-0.0}"
GRPOS_OUTCOME_CONVENTION="${GRPOS_OUTCOME_CONVENTION:-dapo}"
GRPOS_NEGATIVE_ENTROPY_NORM="${GRPOS_NEGATIVE_ENTROPY_NORM:-geometric}"
GRPOS_SHAPE_NEGATIVE="${GRPOS_SHAPE_NEGATIVE:-false}"

# -----------------------------
# Logging / dump（与 FEPO 脚本一致的可选目录）
# -----------------------------
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUTPUT_DIR}/validation_data}"

DUMP_ARGS=()
if [ -n "${ROLLOUT_DATA_DIR}" ]; then
  mkdir -p "${ROLLOUT_DATA_DIR}"
  DUMP_ARGS+=("trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}")
fi
if [ -n "${VALIDATION_DATA_DIR}" ]; then
  mkdir -p "${VALIDATION_DATA_DIR}"
  DUMP_ARGS+=("trainer.validation_data_dir=${VALIDATION_DATA_DIR}")
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo_s \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.grpos_beta1="${GRPOS_BETA1}" \
  algorithm.grpos_beta2="${GRPOS_BETA2}" \
  algorithm.grpos_entropy_clip_low="${GRPOS_ENTROPY_CLIP_LOW}" \
  algorithm.grpos_entropy_clip_high="${GRPOS_ENTROPY_CLIP_HIGH}" \
  algorithm.grpos_success_reward_threshold="${GRPOS_SUCCESS_THRESHOLD}" \
  algorithm.grpos_outcome_convention="${GRPOS_OUTCOME_CONVENTION}" \
  algorithm.grpos_negative_entropy_norm="${GRPOS_NEGATIVE_ENTROPY_NORM}" \
  algorithm.grpos_shape_negative="${GRPOS_SHAPE_NEGATIVE}" \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="['${MATH500_VAL}','${AIME24_VAL}']" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${WARM_UP_RATIO}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "${CUSTOM_REWARD_ARGS[@]}" \
  "${DUMP_ARGS[@]}" \
  "$@"
