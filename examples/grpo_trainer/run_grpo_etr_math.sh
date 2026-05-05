#!/usr/bin/env bash
# GRPO + ETR（Entropy Trend Reward, arXiv:2604.05355）
# - 在 old_log_prob 得到 per-token 熵后，将终端标量奖励改为：错误 R=-1，正确 R=1+λ·R_entropy（见 verl/trainer/ppo/etr_reward.py）。
# - 请使用 outcome GRPO（本脚本默认 algorithm.adv_estimator=grpo_vectorized），勿与 GRPO-S / GTPO 同时开启（避免双重熵塑形）。
#
# 环境变量（可选）：
#   ETR_GAMMA  ETR_LAMBDA  ETR_SEGMENT_MODE(chunk|newline)  ETR_CHUNK_TOKENS

set -euo pipefail

VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B}"
PROJECT_NAME="${PROJECT_NAME:-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_etr_math_qwen3_4b}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/train_outputs/grpo-etr-math}"
mkdir -p "${OUTPUT_DIR}"

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

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"

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

VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"

SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"

ETR_GAMMA="${ETR_GAMMA:-0.9}"
ETR_LAMBDA="${ETR_LAMBDA:-0.1}"
ETR_SEGMENT_MODE="${ETR_SEGMENT_MODE:-chunk}"
ETR_CHUNK_TOKENS="${ETR_CHUNK_TOKENS:-32}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo_vectorized \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  algorithm.etr.enable=True \
  algorithm.etr.gamma="${ETR_GAMMA}" \
  algorithm.etr.lambda_coef="${ETR_LAMBDA}" \
  algorithm.etr.segment_mode="${ETR_SEGMENT_MODE}" \
  algorithm.etr.chunk_tokens="${ETR_CHUNK_TOKENS}" \
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
  "$@"
