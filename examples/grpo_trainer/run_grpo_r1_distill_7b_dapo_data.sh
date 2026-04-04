#!/usr/bin/env bash
# 标准 GRPO：DeepSeek-R1-Distill-Qwen-7B，DAPO 格式数学数据训练，验证集为 MATH-500 + AIME24。
# 硬件：单机 8 卡。训练 batch 与 PPO mini-batch 按你的要求固定。
#
# 数据准备说明：
# 1) 训练集：DAPO 风格 parquet，每条需含 prompt、data_source（建议 math_dapo）、reward_model.ground_truth。
#    可用同目录下 prepare_dapo_style_parquet.py 从 JSONL 转换。
# 2) MATH-500 测试 parquet：列 data_source 需为 HuggingFaceH4/MATH-500（与 default_compute_score 一致），
#    并有 reward_model.ground_truth（字符串或可解析答案）。
# 3) AIME24 测试 parquet：data_source 建议以 aime 开头（如 aime2024），走 math_dapo 判分分支。
#
# 若尚无测试 parquet，可参考 examples/data_preprocess 下脚本生成，或从 HF 拉取后按 README 字段补全。

set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

MODEL_PATH="${MODEL_PATH:-/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/DeepSeek-R1-Distill-Qwen-7B}"

PROJECT_NAME="${PROJECT_NAME:-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dapo_train_math500_aime24_val}"

# Checkpoint 与训练产物本地目录（trainer.default_local_dir）
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/verl_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "${OUTPUT_DIR}"
echo "Checkpoint / trainer output dir: ${OUTPUT_DIR}"

# 训练：DAPO 数据（parquet）
TRAIN_FILES="${TRAIN_FILES:-${HOME}/data/dapo/train.parquet}"

# 验证：两个测试集（OmegaConf 列表字符串）
MATH500_VAL="${MATH500_VAL:-${HOME}/data/math500/test.parquet}"
AIME24_VAL="${AIME24_VAL:-${HOME}/data/aime24/test.parquet}"

# Actor 学习率 warmup：占总训练步的比例（对应 optim.lr_warmup_steps_ratio）；设为 0 关闭
WARM_UP_RATIO="${WARM_UP_RATIO:-0.05}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
