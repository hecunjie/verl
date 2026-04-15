## 采样测试 mc128 vs 随机topk
MAX_SAMPLES=300 \
MAX_NEW_TOKENS=8192 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
NPROC_PER_NODE=8 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MC_M_SAMPLES=128 \
ENTROPY_THRESHOLD=1.0 \
MAX_BRANCH_STEPS=64 \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare \
PROGRESS_ALL_RANKS=0 \
PROGRESS_NESTED=1 \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh


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
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_greedy_path \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh

MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
SELECTION_F_MODE=lookahead_1step \
MAX_BRANCH_STEPS=64 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_lookahead_1step \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh

SELECTION_F_MODE=bucket_group_estimate \
BUCKET_GROUP_ROLLOUTS=16 \
BUCKET_NUM_BINS=100 \
BUCKET_MIN_POINTS_PER_BIN=4 \
MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_bucket_group_estimate \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
SELECTION_F_MODE=bucket_group_estimate \
MAX_BRANCH_STEPS=64 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh

MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_bucket_vs_mc \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_M_SAMPLES_REF=128 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
BUCKET_GROUP_ROLLOUTS=16 \
BUCKET_NUM_BINS=100 \
BUCKET_MIN_POINTS_PER_BIN=2 \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_compare_bias_sign_bucket_vs_mc_sharded.sh

MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_group_mc_vs_bucket_mc_qwen3_4b_base \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_M_SAMPLES_REF=64 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=128 \
BIAS_METRICS_MODE=length_normalized \
BUCKET_GROUP_ROLLOUTS=16 \
BUCKET_PREFIX_KEY_MODE=both \
MATH_EVAL_BACKEND=math_verify \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_compare_bias_sign_bucket_vs_mc_sharded.sh



## Qwen3-4B：高熵步 min-F（MC 续写估计 F），MC=10 与 MC=128 各跑一版（8 卡 vLLM；脚本内会 cd 到 verl 根目录）
## 数据与阈值与 entropy_ce consoles 对齐；OUTPUT_DIR 请按需改
SELECTION_F_MODE=mc \
MC_M_SAMPLES=10 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_4b_instruct_mc10 \
NPROC_PER_NODE=8 \
MAX_SAMPLES=300 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
SAMPLING_TEMPERATURE=1.0 \
SAMPLING_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
PROGRESS_ECHO=1 \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh

SELECTION_F_MODE=mc \
MC_M_SAMPLES=128 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_4b_instruct_mc128 \
NPROC_PER_NODE=8 \
MAX_SAMPLES=300 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
PROGRESS_ECHO=1 \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh

SELECTION_F_MODE=mc \
MC_M_SAMPLES=10 \
MATH_EVAL_BACKEND=math_verify \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-8B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/math500_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_8b_base_mc10_mathverify \
NPROC_PER_NODE=8 \
MAX_SAMPLES=500 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
PROGRESS_ECHO=1 \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh


# 1. 补全 entropies（8 卡示例）
INPUT_JSONL=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/ds_distill_1.5B_bias_correct_wrong_qwen3-4b-base_256/rollouts_archive_merged.jsonl \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/ds_distill_1.5B_bias_correct_wrong_qwen3-4b-base_256/entropy_curve \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
NPROC_PER_NODE=8 \
bash examples/entropy_ce/run_replay_rollouts_archive_entropies_vllm_sharded.sh

# 2. 画正确 vs 错误后缀熵曲线
python3 examples/entropy_ce/analyze_suffix_entropy_curve_bins.py \
  --input /mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/ds_distill_1.5B_bias_correct_wrong_qwen3-4b-base_256/entropy_curve/rollouts_archive_with_entropies_merged.jsonl \
  --output_dir /mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/ds_distill_1.5B_bias_correct_wrong_qwen3-4b-base_256/entropy_curve/suffix_curve_plots \
  --num_bins 100


SELECTION_F_MODE=mc \
MC_M_SAMPLES=10 \
MATH_EVAL_BACKEND=math_verify \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/math500_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_4b_base_mc10_mathverify_boxed \
NPROC_PER_NODE=8 \
MAX_SAMPLES=500 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
PROGRESS_ECHO=1 \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh

SAMPLING_TEMPERATURE=1.0 \
SAMPLING_TOP_P=0.95 \
SELECTION_F_MODE=mc \
MC_M_SAMPLES=10 \
MATH_EVAL_BACKEND=math_verify \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-8B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/math500_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_8b_base_mc10_mathverify_boxed \
NPROC_PER_NODE=8 \
MAX_SAMPLES=500 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
PROGRESS_ECHO=1 \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh