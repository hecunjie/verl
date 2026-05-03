# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Math-Verify 打分。

送入 ``math_metric`` 前可做两步压缩（均可用环境变量关闭），再取最后一个 ``\\boxed{}``：

1. **相邻重复行**：整行 ``strip()`` 后与上一行相同则丢弃（只处理有换行分隔的复读）。
2. **单行内同一字符超长连写**：如 ``靰靰靰靰…``（模型乱码式复读），将连续相同字符压成 1 个（默认至少 20 个才压）。

环境变量：

- ``VERL_MATH_VERIFY_DEDUPE_LINES=0``：关闭 (1)。
- ``VERL_MATH_VERIFY_COLLAPSE_CHAR_RUN_MIN``：触发 (2) 的最短连写长度，默认 ``20``；设为 ``0`` 关闭 (2)。
"""

from __future__ import annotations

import logging
import os
import re

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed

_MATH_VERIFY_LOG_CONFIGURED = False


def _configure_math_verify_logging_once() -> None:
    """Optionally silence noisy internal math_verify/sympy logs.

    Some expressions (e.g. symbolic function forms like ``f(x)=1``) can trigger
    ``NotImplementedError`` inside math_verify's symbolic comparator. The library
    catches those exceptions but logs full tracebacks at ERROR level, which can
    flood distributed worker logs. By default we quiet these internal loggers.
    """
    global _MATH_VERIFY_LOG_CONFIGURED
    if _MATH_VERIFY_LOG_CONFIGURED:
        return
    _MATH_VERIFY_LOG_CONFIGURED = True

    quiet = os.environ.get("VERL_MATH_VERIFY_QUIET_LOG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )
    if not quiet:
        return

    # Keep our outer exception handling behavior unchanged; only reduce noisy internals.
    for name in ("math_verify", "math_verify.grader", "math_verify.metric", "sympy"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _dedupe_consecutive_lines(text: str) -> str:
    """去掉**相邻**重复行（按 ``str.strip()`` 后比较）。

    模型在长推理里常会连续复读同一句/同一段；折叠后可缩短输入、减轻 Math-Verify 负担。
    非相邻重复不在此处理（需要全局去重则要用更重的算法或第三方库，见模块说明）。
    """
    if not text:
        return text
    lines = text.splitlines()
    if len(lines) <= 1:
        return text
    kept: list[str] = []
    prev_key: str | None = None
    for line in lines:
        key = line.strip()
        if key == prev_key:
            continue
        prev_key = key
        kept.append(line)
    return "\n".join(kept)


def _collapse_long_runs_of_same_char(text: str, min_run: int) -> str:
    """将连续出现 ``min_run`` 次及以上的**同一字符**压成 1 个（含 CJK、换行等任意码点）。

    与按行去重互补：乱码常出现在**同一行内**无换行的字符连写，行级去重无法处理。
    """
    if min_run < 2 or not text:
        return text
    # 首字符 + 其余至少 (min_run - 1) 次相同 → 总长 >= min_run
    return re.sub(r"(.)\1{" + str(min_run - 1) + r",}", r"\1", text, flags=re.DOTALL)


def _prepare_model_output_for_verify(model_output: str) -> str:
    """去重（可选）→ 收窄到 ``\\boxed{...}``（若有）。"""
    text = model_output
    if os.environ.get("VERL_MATH_VERIFY_DEDUPE_LINES", "1").strip().lower() not in ("0", "false", "no", ""):
        text = _dedupe_consecutive_lines(text)
    raw_min = os.environ.get("VERL_MATH_VERIFY_COLLAPSE_CHAR_RUN_MIN", "20").strip()
    try:
        run_min = int(raw_min) if raw_min else 0
    except ValueError:
        run_min = 20
    if run_min >= 2:
        text = _collapse_long_runs_of_same_char(text, run_min)
    return _narrow_model_output_for_verify(text)


def _narrow_model_output_for_verify(model_output: str) -> str:
    """Prefer the last ``\\boxed{...}`` span so Math-Verify parses a short LaTeX snippet.

    Long chain-of-thought triggers timeouts more often; extracting the final boxed answer first
    keeps semantics while reducing work inside ``math_metric``.
    """
    try:
        raw = last_boxed_only_string(model_output)
        if raw is None:
            return model_output
        inner = remove_boxed(raw).strip()
        if not inner:
            return model_output
        return "\\boxed{" + inner + "}"
    except Exception:
        return model_output


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    _configure_math_verify_logging_once()
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    pred_for_verify = _prepare_model_output_for_verify(model_output)
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [pred_for_verify])
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass

    return ret_score


def compute_score_with_pred(
    model_output: str, ground_truth: str, timeout_score: float = 0
) -> tuple[float, str | None]:
    """Same scoring as ``compute_score``, plus the extracted prediction string from Math-Verify.

    The second return of ``math_metric`` is ``(golds, preds)`` flattened string lists; we surface
    ``preds`` (joined if multiple) as the prediction text used for verification.
    """
    _configure_math_verify_logging_once()
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    str_preds: tuple[list[str], list[str]] | None = None

    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    pred_for_verify = _prepare_model_output_for_verify(model_output)
    try:
        ret_score, str_preds = verify_func([ground_truth_boxed], [pred_for_verify])
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass

    pred_str: str | None = None
    if str_preds is not None:
        _golds, preds = str_preds
        if preds:
            pred_str = ", ".join(preds) if len(preds) > 1 else preds[0]

    return ret_score, pred_str
