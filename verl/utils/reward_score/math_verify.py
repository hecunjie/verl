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

送入 ``math_metric`` 前可对长输出做**连续行去重**（缓解复读导致的超时），再取最后一个 ``\\boxed{}``。
环境变量 ``VERL_MATH_VERIFY_DEDUPE_LINES=0`` 可关闭去重。
"""

from __future__ import annotations

import os

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed


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


def _prepare_model_output_for_verify(model_output: str) -> str:
    """去重（可选）→ 收窄到 ``\\boxed{...}``（若有）。"""
    text = model_output
    if os.environ.get("VERL_MATH_VERIFY_DEDUPE_LINES", "1").strip().lower() not in ("0", "false", "no", ""):
        text = _dedupe_consecutive_lines(text)
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
