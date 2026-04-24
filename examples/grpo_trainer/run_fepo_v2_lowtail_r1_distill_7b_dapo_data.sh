#!/usr/bin/env bash
# FEPO v2 (low-tail advantage modulation):
# - 方向由原始 A_t 决定
# - 强度由后缀熵率分位 q_t 决定，仅在高熵位点放大 low-tail
#
# 该脚本适配新版 fepo.variant=lowtail_adv。

set -euo pipefail

VERL_ROOT="${VERL_ROOT:-/mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl}"
cd "${VERL_ROOT}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"  # 可在外部注入

MODEL_PATH="${MODEL_PATH:-/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B}"
PROJECT_NAME="${PROJECT_NAME:-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fepo_v2_lowtail_qwen3_4b_dapo}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/qwen3-4b-fepo-v2-lowtail}"

mkdir -p "${OUTPUT_DIR}"
echo "Checkpoint / trainer output dir: ${OUTPUT_DIR}"

TRAIN_FILES="${TRAIN_FILES:-/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet}"
MATH500_VAL="${MATH500_VAL:-/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/math500_test.parquet}"
AIME24_VAL="${AIME24_VAL:-/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet}"
USE_MATH_VERIFY_VAL="${USE_MATH_VERIFY_VAL:-1}"
CUSTOM_REWARD_FUNCTION_PATH="${CUSTOM_REWARD_FUNCTION_PATH:-${VERL_ROOT}/examples/grpo_trainer/math_verify_val_reward.py}"

CUSTOM_REWARD_ARGS=()
if [ "${USE_MATH_VERIFY_VAL}" = "1" ]; then
  CUSTOM_REWARD_ARGS+=("custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH}")
  CUSTOM_REWARD_ARGS+=("custom_reward_function.name=compute_score")
  echo "Validation scorer routing enabled: ${CUSTOM_REWARD_FUNCTION_PATH}"
fi

WARM_UP_RATIO="${WARM_UP_RATIO:-0.05}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ROLLOUT_LOGPROB_MB_PER_GPU="${ROLLOUT_LOGPROB_MB_PER_GPU:-16}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
REF_LOGPROB_MB_PER_GPU="${REF_LOGPROB_MB_PER_GPU:-4}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUTPUT_DIR}/validation_data}"
FEPO_DATA_DIR="${FEPO_DATA_DIR:-${OUTPUT_DIR}/fepo_data}"

# FEPO v2 配置（核心）
FEPO_ENABLE="${FEPO_ENABLE:-true}"
FEPO_VARIANT="${FEPO_VARIANT:-lowtail_adv}"  # lowtail_adv / legacy_mc_bonus
FEPO_H_THRESHOLD="${FEPO_H_THRESHOLD:-2.0}"  # 复用现有熵阈值参数
FEPO_ALPHA="${FEPO_ALPHA:-0.2}"              # m = 1 + alpha * I(q<=beta)
FEPO_BETA="${FEPO_BETA:-0.2}"                # low-tail 分位阈值
FEPO_SUFFIX_MODE="${FEPO_SUFFIX_MODE:-sentence}"   # sentence / full
FEPO_F_SENTENCE_STOP="${FEPO_F_SENTENCE_STOP:-simple}"  # simple / pysbd
FEPO_SENTENCE_MIN_SUFFIX_TOKENS="${FEPO_SENTENCE_MIN_SUFFIX_TOKENS:-5}"
FEPO_SENTENCE_NUM_THREADS="${FEPO_SENTENCE_NUM_THREADS:-8}"
FEPO_SENTENCE_ONLY_HIGH_ENTROPY="${FEPO_SENTENCE_ONLY_HIGH_ENTROPY:-true}"
FEPO_ENTROPY_TOP_P="${FEPO_ENTROPY_TOP_P:-0.95}"  # 1.0=全词表熵，<1 为 topp 截断熵
FEPO_DUMP_FREQ="${FEPO_DUMP_FREQ:-50}"

DUMP_ARGS=()
if [ -n "${ROLLOUT_DATA_DIR}" ]; then
  mkdir -p "${ROLLOUT_DATA_DIR}"
  DUMP_ARGS+=("trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}")
  echo "Rollout data dump enabled: ${ROLLOUT_DATA_DIR}"
fi
if [ -n "${VALIDATION_DATA_DIR}" ]; then
  mkdir -p "${VALIDATION_DATA_DIR}"
  DUMP_ARGS+=("trainer.validation_data_dir=${VALIDATION_DATA_DIR}")
  echo "Validation generations dump enabled: ${VALIDATION_DATA_DIR}"
fi
if [ -n "${FEPO_DATA_DIR}" ]; then
  mkdir -p "${FEPO_DATA_DIR}"
  DUMP_ARGS+=("+trainer.fepo_data_dir=${FEPO_DATA_DIR}")
  echo "FEPO sampled points dump enabled: ${FEPO_DATA_DIR}"
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  +algorithm.fepo.enable="${FEPO_ENABLE}" \
  +algorithm.fepo.variant="${FEPO_VARIANT}" \
  +algorithm.fepo.h_threshold="${FEPO_H_THRESHOLD}" \
  +algorithm.fepo.alpha="${FEPO_ALPHA}" \
  +algorithm.fepo.beta="${FEPO_BETA}" \
  +algorithm.fepo.suffix_mode="${FEPO_SUFFIX_MODE}" \
  +algorithm.fepo.f_sentence_stop="${FEPO_F_SENTENCE_STOP}" \
  +algorithm.fepo.sentence_min_suffix_tokens="${FEPO_SENTENCE_MIN_SUFFIX_TOKENS}" \
  +algorithm.fepo.sentence_num_threads="${FEPO_SENTENCE_NUM_THREADS}" \
  +algorithm.fepo.sentence_only_high_entropy="${FEPO_SENTENCE_ONLY_HIGH_ENTROPY}" \
  +algorithm.fepo.entropy_top_p="${FEPO_ENTROPY_TOP_P}" \
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
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${WARM_UP_RATIO}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
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
  +trainer.fepo_dump_freq="${FEPO_DUMP_FREQ}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "${CUSTOM_REWARD_ARGS[@]}" \
  "${DUMP_ARGS[@]}" \
  "$@"

