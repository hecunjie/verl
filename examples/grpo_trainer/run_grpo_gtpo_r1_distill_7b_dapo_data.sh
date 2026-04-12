#!/usr/bin/env bash
# GTPO（arXiv:2508.04349）：在 VERL GRPO 流程上用动态熵权重分配 token 级 advantage。
# 与 run_grpo_r1_distill_7b_dapo_data.sh 相同，仅 `algorithm.adv_estimator=grpo_gtpo`。
#
# 注意：不要与 rollout_correction 的 bypass_mode 同开（需要 old_log_prob 返回的 per-token entropy）。
#
set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/DeepSeek-R1-Distill-Qwen-7B}"

PROJECT_NAME="${PROJECT_NAME:-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dapo_train_math500_aime24_val_gtpo}"

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/verl_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "${OUTPUT_DIR}"
echo "Checkpoint / trainer output dir: ${OUTPUT_DIR}"

TRAIN_FILES="${TRAIN_FILES:-${HOME}/data/math_rl/dapo_math_17k_processed_train.parquet}"
MATH500_VAL="${MATH500_VAL:-${HOME}/data/math500/test.parquet}"
AIME24_VAL="${AIME24_VAL:-${HOME}/data/aime24/test.parquet}"

WARM_UP_RATIO="${WARM_UP_RATIO:-0.05}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_gtpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="['${MATH500_VAL}','${AIME24_VAL}']" \
    data.train_batch_size=256 \
    data.gen_batch_size=256 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${WARM_UP_RATIO}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    "$@"
