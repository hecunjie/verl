## 采样测试 mc128 vs 随机topk
MAX_SAMPLES=300 MAX_NEW_TOKENS=8192 F_CONTINUATION_MODE=first_sentence F_SENTENCE_MAX_NEW_TOKENS=256 BIAS_METRICS_MODE=length_normalized MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet NPROC_PER_NODE=8 CANDIDATE_TOP_P=0.95 CANDIDATE_MAX_K=5 MC_M_SAMPLES=128 ENTROPY_THRESHOLD=1.0 MAX_BRANCH_STEPS=64 OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare PROGRESS_ALL_RANKS=0 PROGRESS_NESTED=1 PROGRESS_ECHO=1 bash verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh


## 贪婪1代替mc128，greedy baseline, 随机topk
MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
SELECTION_F_MODE=greedy_path \
MAX_BRANCH_STEPS=64 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh