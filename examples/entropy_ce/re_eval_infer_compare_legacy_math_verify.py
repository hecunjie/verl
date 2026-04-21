#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_dapo as math_dapo_score


def _is_math_like_source(data_source: str) -> bool:
    ds = str(data_source or "")
    dsl = ds.lower()
    return (
        ds
        in {
            "math_dapo",
            "math",
            "math_dapo_reasoning",
            "lighteval/MATH",
            "DigitalLearningGmbH/MATH-lighteval",
            "HuggingFaceH4/MATH-500",
        }
        or ds.startswith("aime")
        or ("math500" in dsl)
    )


def _extract_acc(result: Any) -> bool:
    if isinstance(result, dict) and "acc" in result:
        return bool(result["acc"])
    return float(result) > 0.5


def _legacy_math_verify_compute_score(
    model_output: str,
    ground_truth: str,
    timeout_score: float = 0.0,
) -> float:
    """Legacy (pre-2026-04-19) math_verify scoring behavior."""
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )

    ret_score = float(timeout_score)
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    try:
        # Legacy behavior: catch-all; TimeoutException was not singled out.
        ret_score, _ = verify_func([ground_truth_boxed], [str(model_output)])
    except Exception:
        pass
    return float(ret_score)


def evaluate_solution_acc_legacy(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: str,
    math_eval_backend: str,
) -> tuple[bool, dict[str, Any]]:
    ds = str(data_source or "")
    backend = str(math_eval_backend)
    if backend not in {"auto", "math_dapo", "math_verify"}:
        raise ValueError(f"unsupported math_eval_backend: {backend}")

    math_like = _is_math_like_source(ds)

    if math_like and backend in {"auto", "math_dapo"}:
        res = math_dapo_score.compute_score(solution_str, ground_truth, strict_box_verify=False)
        return bool(res.get("acc", False)), {"mode": "math_dapo_minerva", **res}

    if math_like and backend == "math_verify":
        try:
            sc = _legacy_math_verify_compute_score(solution_str, ground_truth)
            ok = bool(float(sc) > 0.5)
            return ok, {"mode": "legacy_math_verify", "score": float(sc)}
        except Exception as e:
            res = math_dapo_score.compute_score(solution_str, ground_truth, strict_box_verify=False)
            return bool(res.get("acc", False)), {
                "mode": "legacy_math_verify_fallback_math_dapo",
                "fallback_reason": f"{type(e).__name__}: {e}",
                **res,
            }

    res = default_compute_score(data_source=ds, solution_str=solution_str, ground_truth=ground_truth)
    return _extract_acc(res), {"mode": "default_compute_score", "raw_result": res}


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else float("nan")


def _response_text_from_record(rec: dict[str, Any], key: str) -> str:
    obj = rec.get(key)
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("response_text", "") or "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate infer_compare_merged.jsonl with legacy math_verify behavior (pre-2026-04-19)."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to infer_compare_merged.jsonl")
    parser.add_argument(
        "--math_eval_backend",
        choices=["auto", "math_dapo", "math_verify"],
        default="math_verify",
        help="Backend routing for re-evaluation; recommend math_verify for this script.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output summary json path. Default: <input_dir>/legacy_reeval_summary.json",
    )
    parser.add_argument(
        "--write_per_sample",
        action="store_true",
        help="Write per-sample re-eval details to <input_dir>/legacy_reeval_per_sample.jsonl",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"input file not found: {in_path}")

    rows: list[dict[str, Any]] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))

    out_dir = in_path.parent
    out_summary = Path(args.output).expanduser().resolve() if args.output else (out_dir / "legacy_reeval_summary.json")
    out_per_sample = out_dir / "legacy_reeval_per_sample.jsonl"

    acc_minf: list[float] = []
    acc_rand: list[float] = []
    acc_greedy: list[float] = []
    paired_improve = 0
    paired_regress = 0
    paired_improve_g = 0
    paired_regress_g = 0
    mode_hist: dict[str, int] = {}

    per_sample_lines: list[str] = []
    for i, rec in enumerate(rows):
        data_source = str(rec.get("data_source", ""))
        ground_truth = str(rec.get("ground_truth", ""))

        text_minf = _response_text_from_record(rec, "result_min_f_mc")
        text_rand = _response_text_from_record(rec, "result_random_sampling")
        if not text_rand:
            text_rand = _response_text_from_record(rec, "result_random_topk")
        text_greedy = _response_text_from_record(rec, "result_greedy_baseline")

        ok_minf, eval_minf = evaluate_solution_acc_legacy(
            data_source=data_source,
            solution_str=text_minf,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )
        ok_rand, eval_rand = evaluate_solution_acc_legacy(
            data_source=data_source,
            solution_str=text_rand,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )
        ok_greedy, eval_greedy = evaluate_solution_acc_legacy(
            data_source=data_source,
            solution_str=text_greedy,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )

        acc_minf.append(1.0 if ok_minf else 0.0)
        acc_rand.append(1.0 if ok_rand else 0.0)
        acc_greedy.append(1.0 if ok_greedy else 0.0)

        if ok_minf and (not ok_rand):
            paired_improve += 1
        if (not ok_minf) and ok_rand:
            paired_regress += 1
        if ok_minf and (not ok_greedy):
            paired_improve_g += 1
        if (not ok_minf) and ok_greedy:
            paired_regress_g += 1

        for ev in (eval_minf, eval_rand, eval_greedy):
            m = str(ev.get("mode", "unknown"))
            mode_hist[m] = int(mode_hist.get(m, 0) + 1)

        if args.write_per_sample:
            out = {
                "sample_index": int(rec.get("sample_index", i)),
                "data_source": data_source,
                "ground_truth": ground_truth,
                "legacy_eval_min_f_mc": {"is_correct": bool(ok_minf), "eval": eval_minf},
                "legacy_eval_random_sampling": {"is_correct": bool(ok_rand), "eval": eval_rand},
                "legacy_eval_greedy_baseline": {"is_correct": bool(ok_greedy), "eval": eval_greedy},
            }
            per_sample_lines.append(json.dumps(out, ensure_ascii=False))

    summary = {
        "num_prompts": int(len(rows)),
        "accuracy_min_f_mc": _mean(acc_minf),
        "accuracy_random_sampling": _mean(acc_rand),
        "accuracy_random_topk": _mean(acc_rand),  # backward-friendly alias
        "accuracy_greedy_baseline": _mean(acc_greedy),
        "accuracy_gain_abs": _mean(acc_minf) - _mean(acc_rand),
        "accuracy_gain_abs_vs_greedy": _mean(acc_minf) - _mean(acc_greedy),
        "paired_improve_count": int(paired_improve),
        "paired_regress_count": int(paired_regress),
        "paired_net_gain": int(paired_improve - paired_regress),
        "paired_improve_count_vs_greedy": int(paired_improve_g),
        "paired_regress_count_vs_greedy": int(paired_regress_g),
        "paired_net_gain_vs_greedy": int(paired_improve_g - paired_regress_g),
        "reeval_backend": str(args.math_eval_backend),
        "eval_mode_histogram": mode_hist,
        "source_file": str(in_path),
    }

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.write_per_sample:
        with open(out_per_sample, "w", encoding="utf-8") as f:
            for line in per_sample_lines:
                f.write(line + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] summary -> {out_summary}")
    if args.write_per_sample:
        print(f"[done] per-sample -> {out_per_sample}")


if __name__ == "__main__":
    main()

