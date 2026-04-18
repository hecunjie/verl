"""Custom reward routing with Math-Verify for math validation sources.

Behavior:
- **Validation** (``extra_info["validate"] is True``): always score with
  ``math_verify`` for every ``data_source`` (all val parquet rows).
- **Training**: only AIME / MATH-500-like ``data_source`` use ``math_verify``;
  others fall back to VERL ``default_compute_score``.

``validate`` is set by reward managers from ``DataProto.meta_info`` during
``ray_trainer._validate`` (see ``naive`` / ``dapo`` reward managers).
"""

from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def _is_math_verify_source(data_source: str) -> bool:
    ds = str(data_source or "")
    return (
        ds
        in {
            "lighteval/MATH",
            "DigitalLearningGmbH/MATH-lighteval",
            "HuggingFaceH4/MATH-500",
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


def _score_math_verify(solution_str: str, ground_truth: str) -> dict[str, Any]:
    try:
        from verl.utils.reward_score import math_verify
    except Exception as e:
        return {
            "score": 0.0,
            "acc": False,
            "pred": None,
            "format_score": 0.0,
            "from_boxed": False,
            "backend": "math_verify_failed",
            "fallback_reason": f"{type(e).__name__}: {e}",
        }

    try:
        score, pred = math_verify.compute_score_with_pred(solution_str, ground_truth)
        score = float(score)
    except Exception as e:
        return {
            "score": 0.0,
            "acc": False,
            "pred": None,
            "format_score": 0.0,
            "from_boxed": False,
            "backend": "math_verify_failed",
            "fallback_reason": f"{type(e).__name__}: {e}",
        }

    return {
        "score": score,
        "acc": bool(score > 0.5),
        "pred": pred,
        "format_score": 0.0,
        "from_boxed": False,
        "backend": "math_verify",
    }


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
