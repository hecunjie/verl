# Method2 偏差项分析执行方案（vLLM + 多卡）

## 1. 本轮确认后的实验目标

围绕你提出的两点，先做**分析实验**（不改训练）：

1. 先标定“每个前缀需要采样多少条后缀，才能让熵估计方差足够小”。
2. 对每个 prompt 采样多条主路径，在同一 prompt 内各选一条**正确**和**错误**路径，再在两条路径上分别抽 top10% 高熵 token，比较偏差项。

实验默认使用 `vLLM` 推理，采用多卡多进程并行提升速度。

---

## 2. 关键定义与可计算形式

定义（与前文一致）：

$$
F(x_{<t}) = H(X_t,\ldots,X_n \mid x_{<t}),\quad
\delta_t(x_t)=F(x_{<t})-F(x_{\le t})
$$

分解式：

$$
\delta_t(x_t)=H_t+\left(\bar{F}_t-F(x_{\le t})\right)
$$

其中偏差项是：

$$
\text{Bias}_t:=\bar{F}_t-F(x_{\le t})
$$

### `F_t` 怎么估计（你问的重点）

可行实现采用“后缀熵和”的 MC 估计：

$$
\hat{F}_M(x_{<t})=\frac{1}{M}\sum_{j=1}^{M}\sum_{s=t}^{n_j} H_s^{(j)}
$$

即：固定前缀后，采样 `M` 条后缀，逐条计算“后缀各位置 token entropy 之和”，再取均值。  
所以你说的“用整个路径后续位置熵求和”是对的；区别只在于建议用多条后缀平均来降方差。

---

## 3. Phase A：先做采样数-方差标定（必须先做）

目标：确定 `M`（每个前缀后缀采样条数）的最小可用值。

## 3.1 标定数据

- 随机抽 `N_calib=50` 个 prompt
- 每个 prompt 先采 `R=8` 条主路径
- 从这些路径中抽取若干候选前缀位置（例如每题 10 个）

## 3.2 方差曲线

对每个候选前缀，分别用：

- `M in {2,4,8,12,16}`

估计 $\hat{F}_M$，记录：

- 样本方差 `Var[\hat{F}_M]`
- 变异系数 `CV = std/mean`
- 95% bootstrap CI 宽度

## 3.3 选型标准（建议）

选最小 `M*` 满足：

1. 相邻增量收益变小：`Var(M)-Var(next M)` 下降幅度 < 10%
2. 中位数 `CV < 0.1`
3. 95% CI 宽度低于预设阈值（例如 < 0.2 nat/token * 剩余长度）

若预算不足，优先保证 `M>=4`。

---

## 4. Phase B：主分析（Correct vs Wrong）

## 4.1 每个 prompt 采样多主路径

- 每个 prompt 采样 `R_main` 条（建议 `R_main=8~16`）
- 用任务判分器标注每条路径 correct / wrong

仅保留“同时存在正确和错误路径”的 prompt。

## 4.2 每个 prompt 选 1 条正确 + 1 条错误路径

建议选法（固定可复现）：

- 正确组：reward 最高且长度在中位附近的一条
- 错误组：reward 最低且长度在中位附近的一条

这样可减弱长度和极端异常路径干扰。

## 4.3 在两条路径上找 top10% 高熵候选点

对每条路径计算 `H_t`，取 top10% 位置作为候选 token 点。

## 4.4 候选点偏差项估计

对每个候选点：

1. 固定前缀 `x_{<t}`
2. 估计 $\hat{F}_{M*}(x_{\le t})`
3. 估计 $\hat{\bar F}_{t,M*}`（可用 top-K 候选 token 加权）
4. 得到
   - `Bias_t = \hat{\bar F}_{t,M*} - \hat F_{M*}(x_{\le t})`
   - `\hat\delta_t = H_t + Bias_t`

---

## 5. 统计输出

按 Correct / Wrong 两组比较：

1. `Bias_t` 分布（均值/中位数/分位数）
2. `\hat\delta_t` 分布
3. `Bias_t / H_t`（后果项相对强度）
4. `P(Bias_t>0)` 与 `P(Bias_t<0)`

检验：

- Mann-Whitney U
- Cliff's delta 或 Cohen's d
- bootstrap 95% CI

补充一个你关心的符号一致性指标：

$$
S_t = \text{sign}(A_t^R)\cdot Bias_t
$$

若逻辑成立，Correct 组的 `S_t` 应整体更大，Wrong 组更小。

---

## 6. 并行执行方案（vLLM + 多卡多进程）

目录内已有脚本：

- `run_entropy_credit_experiment_vllm_sharded.sh`

该脚本是“每 GPU 一个独立 Python 进程 + vLLM 分片参数”模式，适合本实验。

建议参数：

- `NPROC_PER_NODE=<GPU数>`
- `ROLLOUTS_PER_PROMPT=8`（Phase A 可先 8，Phase B 可增到 12/16）
- `METHOD_B_M_SAMPLES` 先用于方差标定阶段扫描
- `PHASE2_MAX_POSITIONS` 控制候选点上限，避免爆算

建议先跑标定，再跑主分析：

1. 标定：小数据 + 扫 `M`
2. 主分析：固定 `M=M*`

---

## 7. 实现落地建议（在本目录）

建议新增两个轻量脚本（可复用现有主脚本逻辑）：

```text
examples/entropy_ce/
├── calibrate_mc_variance.py        # 输出 M-方差曲线与推荐 M*
├── analyze_correct_wrong_bias.py   # 每题选 1 正确 + 1 错误并做统计
```

若暂不新增脚本，也可先在 `entropy_credit_experiment.py` 增加两个 mode：

- `--analysis_mode calibrate_m`
- `--analysis_mode correct_wrong_pair`

---

## 8. 最小可行配置（先跑通）

- `N_calib=50`, `R=8`, `M grid={2,4,8,12}`
- 主分析 `N_main=300`, `R_main=8`, `top10%` 候选
- `K=3`（估计 $\bar F_t$ 的 top-K）
- `M=M*`（由标定得出，预期在 4~8）

---

## 9. 成功标准

1. 能稳定得到 `M*`（方差曲线有拐点）
2. Correct/Wrong 组在 `Bias_t` 或 `S_t` 上有统计显著差异
3. 结果在不同采样种子下趋势一致

若 2/3 不成立，说明 Method2 偏差项暂不适合作为主信用权重，只能保留为辅助信号。
