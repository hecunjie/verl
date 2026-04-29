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
MC_M_SAMPLES=64 \
MATH_EVAL_BACKEND=math_verify \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/DeepSeek-R1-Distill-Qwen-7B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_deepseek_r1_distill_qwen_7b_mc64_aime24_test1 \
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

SEED=123 \
SAMPLING_TEMPERATURE=1.0 \
SAMPLING_TOP_P=0.95 \
SELECTION_F_MODE=mc \
MC_M_SAMPLES=10 \
MATH_EVAL_BACKEND=math_verify \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_4b_base_mc10_aime24_test1 \
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

SEED=42 \
SAMPLING_TEMPERATURE=1.0 \
SAMPLING_TOP_P=0.95 \
SELECTION_F_MODE=mc \
MC_M_SAMPLES=10 \
MATH_EVAL_BACKEND=math_verify \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_4b_instruct_mc10_aime24_test11 \
NPROC_PER_NODE=8 \
MAX_SAMPLES=500 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_SENTENCE_STOP=simple \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
PROGRESS_ECHO=1 \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh

SAMPLING_TEMPERATURE=1.0 \
SAMPLING_TOP_P=0.95 \
SELECTION_F_MODE=mc_max \
MC_M_SAMPLES=32 \
MATH_EVAL_BACKEND=math_verify \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_f_mc_compare_qwen3_4b_instruct_mcmax10_aime24_test11 \
NPROC_PER_NODE=8 \
MAX_SAMPLES=500 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_SENTENCE_STOP=simple \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
BIAS_METRICS_MODE=length_normalized \
PROGRESS_ECHO=1 \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_topk_f_mc_compare_vllm_sharded.sh

# 用 Reward Model 对 MC(first_sentence) 续写打分：每个候选 token 采样 MC_M_SAMPLES 条续写，按 RM 均分选最好 token
SEED=42 \
SAMPLING_TEMPERATURE=1.0 \
SAMPLING_TOP_P=0.95 \
SELECTION_F_MODE=rm_score_mc \
MC_M_SAMPLES=10 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
F_SENTENCE_STOP=simple \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=256 \
RM_MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/reward_models/Skywork-Reward-Llama-3.1-8B \
RM_MODEL_DEVICE=cuda \
RM_MODEL_MAX_LENGTH=8192 \
RM_RESPONSE_TAIL_TOKENS=8192 \
MATH_EVAL_BACKEND=math_verify \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/infer_topk_rm_score_mc_compare_qwen3_4b_instruct_mc10_aime24_rm8k \
NPROC_PER_NODE=4 \
GPUS_PER_PROCESS=2 \
RM_GPU_LOCAL_INDEX=1 \
MAX_SAMPLES=500 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
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

# 对比实验：在分叉点改为 reward model 选择最优 token（其余保持一致）
MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_reward_skywork_select_qwen3_4b \
NPROC_PER_NODE=4 \
GPUS_PER_PROCESS=2 \
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
BRANCH_TOKEN_SELECTOR=reward_model \
RM_SCORE_BACKEND=open_source_rm \
RM_MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/reward_models/Skywork-Reward-Llama-3.1-8B \
RM_MODEL_DEVICE=cuda \
RM_MODEL_MAX_LENGTH=4096 \
RM_SELECT_DECODE_MODE=greedy \
RM_SELECT_MAX_NEW_TOKENS=256 \
RM_SELECT_TIE_BREAK=candidate_prob \
RM_MC_COMPARE=1 \
MC_TOKEN_SELECT_OBJECTIVE=min_f \
MATH_EVAL_BACKEND=math_verify \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_compare_bias_sign_bucket_vs_mc_sharded.sh

# 对比实验（速度优先 RM baseline）：deberta + 更短候选补全，便于和 future entropy 做同预算对比
MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_reward_select_deberta_qwen3_4b \
NPROC_PER_NODE=4 \
GPUS_PER_PROCESS=2 \
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
BRANCH_TOKEN_SELECTOR=reward_model \
RM_SCORE_BACKEND=open_source_rm \
RM_MODEL_PATH=OpenAssistant/reward-model-deberta-v3-large-v2 \
RM_MODEL_DEVICE=cuda \
RM_MODEL_MAX_LENGTH=4096 \
RM_SELECT_DECODE_MODE=greedy \
RM_SELECT_MAX_NEW_TOKENS=128 \
RM_SELECT_TIE_BREAK=candidate_prob \
RM_MC_COMPARE=1 \
MC_TOKEN_SELECT_OBJECTIVE=min_f \
MATH_EVAL_BACKEND=math_verify \
PROGRESS_ECHO=1 \
bash verl/examples/entropy_ce/run_compare_bias_sign_bucket_vs_mc_sharded.sh

# 同预算对比（future entropy baseline）：与上面 deberta 实验保持同一候选补全长度预算
MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_lookahead1step_budget128_qwen3_4b \
NPROC_PER_NODE=4 \
GPUS_PER_PROCESS=2 \
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
RM_SCORE_BACKEND=open_source_rm \
RM_MODEL_PATH=OpenAssistant/reward-model-deberta-v3-large-v2 \
RM_MODEL_DEVICE=cuda \
RM_MODEL_MAX_LENGTH=4096 \
RM_SELECT_DECODE_MODE=greedy \
RM_SELECT_MAX_NEW_TOKENS=128 \
RM_SELECT_TIE_BREAK=candidate_prob \
RM_MC_COMPARE=1 \
MC_TOKEN_SELECT_OBJECTIVE=min_f \
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

# 首句 f_real + 下一句熵率 f_bar：采样点之后到首句末为 f_real，紧随其后的下一句为 f_bar（需 first_sentence + real_path）
MAX_SAMPLES=300 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/dapo_math_17k_processed_train.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_first_next_sentence_mc_qwen3_4b \
NPROC_PER_NODE=8 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1.0 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=5 \
MAX_BRANCH_STEPS=64 \
MC_M_SAMPLES_REF=64 \
MC_TEMPERATURE=1.0 \
MC_TOP_P=0.95 \
BIAS_METRICS_MODE=length_normalized \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=128 \
F_SENTENCE_STOP=simple \
BUCKET_GROUP_ROLLOUTS=1 \
FBAR_MODE=first_next_sentence \
BRANCH_TOKEN_SELECTOR=real_path \
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
  --input /mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_trend_mc_qwen3_4b_16k_rollouts/sign_compare_merged.jsonl \
  --pr_precision_relative_gap_frac 0.4 \
  --pr_precision_min_abs_gap 0.3 \
  --pr_precision_min_f_bar 0.3 \
  --pr_recall_relative_gap_frac 0.2 \
  --pr_recall_min_abs_gap 0.1 > /mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_trend_mc_qwen3_4b_16k_rollouts/sign_compare_filtered_metrics.log
python3 examples/entropy_ce/analyze_sign_compare_filtered_metrics.py \
  --input /mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_single_local40_mc_qwen3_4b/sign_compare_merged.jsonl \
  --pr_precision_relative_gap_frac 0.4 \
  --pr_precision_min_abs_gap 0.3 \
  --pr_precision_min_f_bar 0.3 \
  --pr_recall_relative_gap_frac 0.2 \
  --pr_recall_min_abs_gap 0.1 > /mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/compare_bias_sign_single_local40_mc_qwen3_4b/sign_compare_filtered_metrics.log

# MODE=greedy \




MODE=greedy \

MODE=sampling \
NUM_SAMPLES_PER_PROMPT=32 \
PASS_K_SMALL=4 \
PASS_K_LARGE=32 \
VLLM_REQUEST_BATCH_CHUNK_MC=256 \
SAMPLING_TEMPERATURE=1.0 \
SAMPLING_TOP_P=0.95 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B-base \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/qwen3-4b-base_passk_32/sampling_aime24 \
NPROC_PER_NODE=8 \
MAX_SAMPLES=500 \
MAX_NEW_TOKENS=8192 \
MATH_EVAL_BACKEND=math_verify \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_passk_by_mode_vllm_sharded.sh

MODE=min_f_mc \
MINF_NONBRANCH_MODE=greedy \
NUM_SAMPLES_PER_PROMPT=32 \
PASS_K_SMALL=4 \
PASS_K_LARGE=32 \
SAMPLING_TEMPERATURE=1.0 \
SAMPLING_TOP_P=0.95 \
SELECTION_F_MODE=mc \
MC_M_SAMPLES=10 \
MC_TEMPERATURE=1 \
MC_TOP_P=0.95 \
VLLM_REQUEST_BATCH_CHUNK_MC=256 \
MODEL_PATH=/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen3-4B \
INPUT_DATA=/mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/grpo/aime2024_test.parquet \
OUTPUT_DIR=/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/qwen3-4b_passk_32/minfmc_nonbranch_greedy_aime24_batch_mc10_entropy1.0_maxk10 \
NPROC_PER_NODE=8 \
MAX_SAMPLES=500 \
MAX_NEW_TOKENS=8192 \
ENTROPY_THRESHOLD=1 \
CANDIDATE_TOP_P=0.95 \
CANDIDATE_MAX_K=10 \
MAX_BRANCH_STEPS=64 \
F_CONTINUATION_MODE=first_sentence \
F_SENTENCE_MAX_NEW_TOKENS=128 \
BIAS_METRICS_MODE=length_normalized \
MATH_EVAL_BACKEND=math_verify \
bash /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/verl/examples/entropy_ce/run_infer_passk_by_mode_vllm_sharded.sh \
  --minf_sample_parallel_batch 32


python3 examples/entropy_ce/recompute_passk_metrics_from_merged.py \
  --input /mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/qwen3-4b_passk_32/minfmc_nonbranch_greedy_aime24_batch/passk_mode_merged.jsonl \
  --k_small 4 \
  --k_large 32 \
  --mean_n 32