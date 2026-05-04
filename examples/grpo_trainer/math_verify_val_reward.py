"""Custom reward routing with Math-Verify for math validation sources.

Behavior:
- **Validation** (``extra_info["validate"] is True``): always score with
  ``math_verify`` for every ``data_source`` (all val parquet rows).
- **Training**: only AIME / MATH-500-like ``data_source`` use ``math_verify``;
  others fall back to VERL ``default_compute_score``.

当 ``math_verify`` 给出非正分（含超时/内部比较失败等常见 0 分）而 ``math_dapo`` 判定正确时，
自动回退到 ``math_dapo`` 的结果（见 ``VERL_MATH_VERIFY_FALLBACK_MATH_DAPO``），减少符号求解失败带来的假阴性。

``validate`` is set by reward managers from ``DataProto.meta_info`` during
``ray_trainer._validate`` (see ``naive`` / ``dapo`` reward managers).
"""

from __future__ import annotations

import math
import os
from typing import Any

from verl.utils.reward_score import default_compute_score

# 与 reward 展平后写入 non_tensor_batch 的字段一致；避免部分样本多键导致多 worker concat 断言失败。
_MATH_VERIFY_REWARD_EXTRA_KEYS = ("fallback_reason", "math_verify_score", "math_verify_pred")


def _is_math_verify_source(data_source: str) -> bool:
    ds = str(data_source or "")
    return (
        ds
        in {
            "lighteval/MATH",
            "DigitalLearningGmbH/MATH-lighteval",
            "HuggingFaceH4/MATH-500",
            "math-ai/amc23",
            "math-ai/olympiadbench",
            "svc-huggingface/minerva-math",
        }
        or ds.startswith("aime")
        or "math500" in ds.lower()
    )


def _score_with_default(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float | dict[str, Any]:
    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def _with_stable_reward_keys(d: dict[str, Any]) -> dict[str, Any]:
    """保证每条样本 reward dict 含相同键，便于 agent_loop 各 worker DataProto.concat。"""
    out = dict(d)
    for k in _MATH_VERIFY_REWARD_EXTRA_KEYS:
        out.setdefault(k, None)
    return out


def _fallback_math_dapo(solution_str: str, ground_truth: str, *, reason: str) -> dict[str, Any]:
    from verl.utils.reward_score import math_dapo

    d = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=False)
    sc = float(d.get("score", 0.0))
    return _with_stable_reward_keys(
        {
            "score": sc,
            "acc": bool(d.get("acc", sc > 0.5)),
            "pred": d.get("pred"),
            "format_score": float(d.get("format_score", 0.0)),
            "from_boxed": bool(d.get("from_boxed", False)),
            "backend": "math_verify_fallback_math_dapo",
            "fallback_reason": reason,
            "math_verify_score": None,
            "math_verify_pred": None,
        }
    )


def _score_math_verify(solution_str: str, ground_truth: str) -> dict[str, Any]:
    try:
        from verl.utils.reward_score import math_verify
    except Exception as e:
        return _fallback_math_dapo(solution_str, ground_truth, reason=f"import_math_verify:{type(e).__name__}: {e}")

    fb = os.environ.get("VERL_MATH_VERIFY_FALLBACK_MATH_DAPO", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )

    try:
        score, pred = math_verify.compute_score_with_pred(solution_str, ground_truth)
        score = float(score)
    except Exception as e:
        return _fallback_math_dapo(solution_str, ground_truth, reason=f"math_verify_exception:{type(e).__name__}: {e}")

    if fb and (not math.isfinite(score) or score <= 0.5):
        try:
            from verl.utils.reward_score import math_dapo

            d = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=False)
            if bool(d.get("acc", False)):
                sc = float(d.get("score", 0.0))
                return _with_stable_reward_keys(
                    {
                        "score": sc,
                        "acc": True,
                        "pred": d.get("pred"),
                        "format_score": float(d.get("format_score", 0.0)),
                        "from_boxed": bool(d.get("from_boxed", False)),
                        "backend": "math_verify_fallback_math_dapo",
                        "fallback_reason": "math_verify_nonpositive_but_math_dapo_correct",
                        "math_verify_score": float(score),
                        "math_verify_pred": pred,
                    }
                )
        except Exception:
            pass

    return _with_stable_reward_keys(
        {
            "score": score,
            "acc": bool(score > 0.5),
            "pred": pred,
            "format_score": 0.0,
            "from_boxed": False,
            "backend": "math_verify",
            "fallback_reason": None,
            "math_verify_score": score,
            "math_verify_pred": pred,
        }
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float | dict[str, Any]:
    extra_info = extra_info or {}
    # Validation: all val samples use math_verify regardless of data_source tag.
    if bool(extra_info.get("validate", False)):
        return _score_math_verify(solution_str, ground_truth)

    if not _is_math_verify_source(data_source):
        return _score_with_default(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    return _score_math_verify(solution_str, ground_truth)
