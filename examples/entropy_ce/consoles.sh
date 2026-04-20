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
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-8B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_8b_instruct_mc10_aime24 \
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
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_4b_instruct_mc10_aime24_test7 \
NPROC_PER_NODE=8 \
MAX_SAMPLES=2000 \
MAX_NEW_TOKENS=10240 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=81 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
PROGRESS_ECHO=1 \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh


MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_trend_mc_qwen3_4b_16k_rollouts \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_M_SAMPLES_REF=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=128 \
F_SENTENCE_STOP=simple \
BUCKET_GROUP_ROLLOUTS=16 \
MATH_EVAL_BACKEND=math_verify \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_compare_bias_sign_bucket_vs_mc_sharded.sh

MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_single_local40_mc_qwen3_4b \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_M_SAMPLES_REF=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=128 \
F_SENTENCE_STOP=simple \
BUCKET_GROUP_ROLLOUTS=1 \
FBAR_MODE=single_local \
LOCAL_WINDOW_LEFT_TOKENS=20 \
LOCAL_WINDOW_RIGHT_TOKENS=20 \
MATH_EVAL_BACKEND=math_verify \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_compare_bias_sign_bucket_vs_mc_sharded.sh

# 方案 1：候选 token 上 1-step lookahead 下一步熵 H(next|prefix+c) 作为 f_bar / f_real，与 MC 的 sign(f_bar_mc - f_real_mc) 对齐统计
MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_lookahead1step_mc_qwen3_4b \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_M_SAMPLES_REF=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=128 \
F_SENTENCE_STOP=simple \
BUCKET_GROUP_ROLLOUTS=1 \
FBAR_MODE=lookahead_1step \
LOCAL_WINDOW_LEFT_TOKENS=20 \
LOCAL_WINDOW_RIGHT_TOKENS=20 \
MATH_EVAL_BACKEND=math_verify \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_compare_bias_sign_bucket_vs_mc_sharded.sh

# 方案 1 扩展：2-step lookahead，先在 prefix+c 下按 top-p 取下一步分支，再对下一步后的熵做期望
MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_lookahead2step_mc_qwen3_4b \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_M_SAMPLES_REF=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=128 \
F_SENTENCE_STOP=simple \
BUCKET_GROUP_ROLLOUTS=1 \
FBAR_MODE=lookahead_2step \
LOCAL_WINDOW_LEFT_TOKENS=20 \
LOCAL_WINDOW_RIGHT_TOKENS=20 \
MATH_EVAL_BACKEND=math_verify \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_compare_bias_sign_bucket_vs_mc_sharded.sh

# 同一次 run：MC 参考用 128 条估计 sign(f_bar-f_real)，再采 1 条 MC 估计同符号，在 sign_compare_summary.json 里看 mc_sign_compare_vs_ref_agreement（M=1 与 M=128 符号一致率）；lookahead 仍为 2-step
MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_mc128_vs_mc1_qwen3_4b \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_M_SAMPLES_REF=128 \
MC_M_SAMPLES_COMPARE=1 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=96 \
F_SENTENCE_STOP=simple \
BUCKET_GROUP_ROLLOUTS=1 \
FBAR_MODE=lookahead_2step \
LOCAL_WINDOW_LEFT_TOKENS=20 \
LOCAL_WINDOW_RIGHT_TOKENS=20 \
MATH_EVAL_BACKEND=math_verify \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_compare_bias_sign_bucket_vs_mc_sharded.sh

# 跑完后：analyze 脚本里 PR 准召只应用你显式传入的 --pr_precision_* / --pr_recall_*；未传的过滤条件一律不用
python3 examples/entropy_ce/analyze_sign_compare_filtered_metrics.py \
  --input /mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_mc128_vs_mc1_qwen3_4b/sign_compare_merged.jsonl \
  --pr_precision_min_abs_gap 0.08 \
  --pr_recall_min_abs_gap 0.08 > /mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_mc128_vs_mc1_qwen3_4b/sign_compare_filtered_metrics.log

    --pr_precision_relative_gap_frac 0.3 \
  --pr_recall_relative_gap_frac 0.3 \