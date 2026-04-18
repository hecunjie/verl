"""Custom reward routing with Math-Verify for math validation sources.

Behavior:
- AIME / MATH-500-like sources: score with `verl.utils.reward_score.math_verify`.
- Other sources: fall back to VERL `default_compute_score`.

This keeps training-set behavior close to default when train data_source is
`math_dapo`, while allowing validation sets such as AIME/MATH-500 to use
Math-Verify.
"""

from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def _is_math_verify_source(data_source: str) -> bool:
    ds = str(data_source or "")
    return (
        ds in {
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


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float | dict[str, Any]:
    if not _is_math_verify_source(data_source):
        return _score_with_default(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    try:
        from verl.utils.reward_score import math_verify
    except Exception as e:
        return {
            "score": 0.0,
            "acc": False,
            "backend": "math_verify_failed",
            "fallback_reason": f"{type(e).__name__}: {e}",
        }

    try:
        score = float(math_verify.compute_score(solution_str, ground_truth))
    except Exception as e:
        return {
            "score": 0.0,
            "acc": False,
            "backend": "math_verify_failed",
            "fallback_reason": f"{type(e).__name__}: {e}",
        }

    return {
        "score": score,
        "acc": bool(score > 0.5),
        "backend": "math_verify",
    }
