#!/usr/bin/env bash
# 启动入口：在 VERL 仓库根目录调用 submit_grpo_r1_distill_7b_dapo_ray.sh，向已有 Ray 集群提交训练 Job。
#
# 前置条件：
#   - 本机已安装 ray CLI，且能访问 Ray Dashboard / Job 服务（默认端口 8265）
#   - 集群地址与账号环境一致（多机需共享 OUTPUT_DIR / 数据路径等，见 run_grpo 脚本）
#
# 用法（在任意目录）:
#   bash examples/grpo_trainer/launch_submit_grpo_r1_distill_7b_dapo_ray.sh
#   bash examples/grpo_trainer/launch_submit_grpo_r1_distill_7b_dapo_ray.sh trainer.experiment_name=my_exp
#
set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

# ========== 按需修改（也可用当前 shell 已 export 的变量覆盖）==========
# Ray Job 提交地址：Dashboard 的 http://<host>:8265
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"

# W&B（可选；不设 WANDB_API_KEY 时 submit 脚本会把 online 降为 offline）
# export WANDB_API_KEY="..."
# export WANDB_PROJECT="verl"
# export WANDB_ENTITY=""
# export WANDB_MODE="online"

# vLLM / NCCL（可选，与 submit 脚本默认一致）
# export VLLM_USE_V1=1
# export NCCL_DEBUG=WARN

echo "[launch] VERL_ROOT=${VERL_ROOT}"
echo "[launch] RAY_ADDRESS=${RAY_ADDRESS}"

exec bash examples/grpo_trainer/submit_grpo_r1_distill_7b_dapo_ray.sh "$@"
