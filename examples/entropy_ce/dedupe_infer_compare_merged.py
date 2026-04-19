#!/usr/bin/env python3
"""Deduplicate ``infer_compare_merged.jsonl`` by ``sample_index`` and recompute summary metrics.

Use when merge artifacts contain duplicate ``sample_index`` rows (e.g. reruns, manual merges).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _mean(xs: list[float]) -> float:
    import numpy as np

    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else float("nan")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dedupe_by_sample_index(rows: list[dict[str, Any]], *, keep: str) -> tuple[list[dict[str, Any]], int]:
    """Return deduplicated rows and number of dropped duplicate lines."""
    if keep == "first":
        seen: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        for row in rows:
            idx = int(row["sample_index"])
            if idx not in seen:
                seen[idx] = row
                order.append(idx)
        dropped = len(rows) - len(seen)
        return [seen[i] for i in order], dropped
    if keep == "last":
        seen = {}
        for row in rows:
            idx = int(row["sample_index"])
            seen[idx] = row
        dropped = len(rows) - len(seen)
        merged = [seen[i] for i in sorted(seen.keys())]
        return merged, dropped
    raise ValueError(f"keep must be 'first' or 'last', got {keep!r}")


def build_summary(merged: list[dict[str, Any]]) -> dict[str, Any]:
    """Match ``infer_topk_f_mc_compare.py`` rank-0 summary (without requiring argparse config)."""
    n = len(merged)
    if n == 0:
        return {"num_prompts": 0, "note": "empty after dedupe"}

    paired_improve = 0
    paired_regress = 0
    paired_improve_g = 0
    paired_regress_g = 0
    for x in merged:
        ok_minf = bool(x["result_min_f_mc"]["is_correct"])
        ok_rand = bool(x.get("result_random_sampling", x.get("result_random_topk", {})).get("is_correct", False))
        ok_greedy = bool(x["result_greedy_baseline"]["is_correct"])
        paired_improve += int(ok_minf and not ok_rand)
        paired_regress += int((not ok_minf) and ok_rand)
        paired_improve_g += int(ok_minf and not ok_greedy)
        paired_regress_g += int((not ok_minf) and ok_greedy)

    acc_minf_all = [1.0 if x["result_min_f_mc"]["is_correct"] else 0.0 for x in merged]
    acc_rand_all = [
        1.0
        if (x.get("result_random_sampling", x.get("result_random_topk", {})).get("is_correct", False))
        else 0.0
        for x in merged
    ]
    acc_greedy_all = [1.0 if x["result_greedy_baseline"]["is_correct"] else 0.0 for x in merged]
    branch_minf_all = [float(x["result_min_f_mc"]["num_branch_steps"]) for x in merged]
    branch_rand_all = [
        float(x.get("result_random_sampling", x.get("result_random_topk", {})).get("num_branch_steps", 0.0))
        for x in merged
    ]
    branch_greedy_all = [float(x["result_greedy_baseline"]["num_branch_steps"]) for x in merged]
    len_minf_all = [float(x["result_min_f_mc"]["response_len_tokens"]) for x in merged]
    len_rand_all = [
        float(x.get("result_random_sampling", x.get("result_random_topk", {})).get("response_len_tokens", 0.0))
        for x in merged
    ]
    len_greedy_all = [float(x["result_greedy_baseline"]["response_len_tokens"]) for x in merged]

    first = merged[0]
    config = {
        "entropy_threshold": float(first.get("entropy_threshold", 0.0)),
        "candidate_top_p": float(first.get("candidate_top_p", 0.0)),
        "candidate_max_k": int(first.get("candidate_max_k", 0)),
        "max_new_tokens": int(first.get("max_new_tokens", 0)),
        "max_branch_steps": int(first.get("max_branch_steps", 0)),
        "mc_m_samples": int(first.get("mc_m_samples", 0)),
        "selection_f_mode": str(first.get("selection_f_mode", "")),
        "math_eval_backend": str(first.get("math_eval_backend", "")),
        "bucket_group_rollouts": int(first.get("bucket_group_rollouts", 0)),
        "bucket_num_bins": int(first.get("bucket_num_bins", 0)),
        "bucket_min_points_per_bin": int(first.get("bucket_min_points_per_bin", 0)),
        "mc_temperature": float(first.get("mc_temperature", 0.0)),
        "mc_top_p": float(first.get("mc_top_p", 0.0)),
        "sampling_temperature": float(first.get("sampling_temperature", 0.0)),
        "sampling_top_p": float(first.get("sampling_top_p", 0.0)),
        "f_continuation_mode": str(first.get("f_continuation_mode", "")),
        "f_sentence_max_new_tokens": int(first.get("f_sentence_max_new_tokens", 0)),
        "f_sentence_stop": str(first.get("f_sentence_stop", "")),
        "bias_metrics_mode": str(first.get("bias_metrics_mode", "")),
    }

    return {
        "num_prompts": int(n),
        "accuracy_min_f_mc": _mean(acc_minf_all),
        "accuracy_random_sampling": _mean(acc_rand_all),
        "accuracy_random_topk": _mean(acc_rand_all),
        "accuracy_greedy_baseline": _mean(acc_greedy_all),
        "accuracy_gain_abs": _mean(acc_minf_all) - _mean(acc_rand_all),
        "accuracy_gain_abs_vs_greedy": _mean(acc_minf_all) - _mean(acc_greedy_all),
        "paired_improve_count": int(paired_improve),
        "paired_regress_count": int(paired_regress),
        "paired_net_gain": int(paired_improve - paired_regress),
        "paired_improve_count_vs_greedy": int(paired_improve_g),
        "paired_regress_count_vs_greedy": int(paired_regress_g),
        "paired_net_gain_vs_greedy": int(paired_improve_g - paired_regress_g),
        "avg_branch_steps_min_f_mc": _mean(branch_minf_all),
        "avg_branch_steps_random_sampling": _mean(branch_rand_all),
        "avg_branch_steps_random_topk": _mean(branch_rand_all),
        "avg_branch_steps_greedy_baseline": _mean(branch_greedy_all),
        "avg_response_len_min_f_mc": _mean(len_minf_all),
        "avg_response_len_random_sampling": _mean(len_rand_all),
        "avg_response_len_random_topk": _mean(len_rand_all),
        "avg_response_len_greedy_baseline": _mean(len_greedy_all),
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedupe infer_compare_merged.jsonl by sample_index and rescore.")
    parser.add_argument(
        "input_jsonl",
        type=str,
        help="Path to infer_compare_merged.jsonl",
    )
    parser.add_argument(
        "--keep",
        choices=("first", "last"),
        default="first",
        help="When duplicate sample_index: keep first or last row in file order (default: first).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Write deduped rows here (default: <input_stem>_deduped.jsonl next to input).",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=None,
        help="Write summary JSON here (default: <input_stem>_summary_deduped.json next to input).",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print summary to stdout.")
    args = parser.parse_args()

    inp = Path(args.input_jsonl).expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"not a file: {inp}")

    rows = load_jsonl(inp)
    merged, dropped = dedupe_by_sample_index(rows, keep=args.keep)

    stem = inp.stem
    out_jsonl = Path(args.output_jsonl) if args.output_jsonl else inp.with_name(f"{stem}_deduped.jsonl")
    out_summary = Path(args.output_summary) if args.output_summary else inp.with_name(f"{stem}_summary_deduped.json")

    summary = build_summary(merged)
    summary["dedupe_meta"] = {
        "input_lines": len(rows),
        "unique_sample_index": len(merged),
        "dropped_duplicate_lines": dropped,
        "keep": args.keep,
    }

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nWrote {out_jsonl}", file=__import__("sys").stderr)
        print(f"Wrote {out_summary}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
