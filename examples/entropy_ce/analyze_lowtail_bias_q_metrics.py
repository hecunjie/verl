#!/usr/bin/env python3
"""Aggregate bias>0 accuracy and recall from low-tail bias validation JSONL.

Reads records produced by collect_lowtail_bias_validation.py (fields: q, bias,
f_suffix_rate, suffix_tokens_used, etc.).

Definitions (aligned with collect script summary):
- q: rank percentile of length-bucket-normalized suffix entropy rate; low-tail
  uses small q (q <= cutoff).
- Accuracy: P(bias > 0 | q <= cutoff) among tokens with finite q in the tail set.
- Recall: among tokens with bias > 0 and |bias| > tau, fraction with q <= cutoff.

When loading multiple shard files (e.g. lowtail_bias_points_rank*.jsonl), use
--recompute-q (default) so q is computed on the union, matching merged
lowtail_bias_points.jsonl behavior.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _length_bucket_normalize(
    records: list[dict[str, Any]],
    *,
    bin_width: int,
    min_count: int,
    eps: float = 1e-6,
) -> None:
    buckets: dict[int, list[float]] = {}
    all_vals: list[float] = []
    bw = max(1, int(bin_width))
    for r in records:
        x = float(r.get("f_suffix_rate", float("nan")))
        l = int(r.get("suffix_tokens_used", 0) or 0)
        if not np.isfinite(x) or l <= 0:
            continue
        bid = (l - 1) // bw
        buckets.setdefault(bid, []).append(x)
        all_vals.append(x)
    global_mean = float(np.mean(all_vals)) if all_vals else 0.0
    global_std = float(np.std(all_vals)) if len(all_vals) > 1 else 0.0
    stats: dict[int, tuple[float, float, int]] = {}
    for bid, xs in buckets.items():
        arr = np.asarray(xs, dtype=np.float64)
        stats[bid] = (float(np.mean(arr)), float(np.std(arr)), int(arr.size))

    for r in records:
        x = float(r.get("f_suffix_rate", float("nan")))
        l = int(r.get("suffix_tokens_used", 0) or 0)
        if not np.isfinite(x) or l <= 0:
            r["f_suffix_rate_norm"] = float("nan")
            r["suffix_len_bucket"] = None
            r["suffix_len_bucket_count"] = 0
            r["suffix_len_bucket_fallback"] = True
            continue
        bid = (l - 1) // bw
        mean, std, cnt = stats.get(bid, (global_mean, global_std, 0))
        fallback = cnt < int(min_count)
        if fallback:
            mean, std, cnt = global_mean, global_std, len(all_vals)
        r["suffix_len_bucket"] = int(bid)
        r["suffix_len_bucket_count"] = int(cnt)
        r["suffix_len_bucket_fallback"] = bool(fallback)
        r["f_suffix_rate_norm"] = float((x - mean) / math.sqrt(std * std + eps))


def _assign_q(records: list[dict[str, Any]], key: str = "f_suffix_rate_norm") -> None:
    valid = [(i, float(r.get(key, float("nan")))) for i, r in enumerate(records)]
    valid = [(i, x) for i, x in valid if np.isfinite(x)]
    valid_sorted = sorted(valid, key=lambda z: z[1])
    n = len(valid_sorted)
    for rank, (i, _x) in enumerate(valid_sorted):
        records[i]["q"] = float(rank / (n - 1)) if n > 1 else 0.0
    for r in records:
        if "q" not in r:
            r["q"] = float("nan")


def _load_jsonl_paths(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _expand_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in inputs:
        p = Path(s).expanduser()
        if any(ch in str(p) for ch in "*?["):
            out.extend(Path(x) for x in sorted(glob_module.glob(str(p))))
        else:
            out.append(p.resolve())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        nargs="+",
        required=True,
        help="One or more JSONL files (e.g. merged lowtail_bias_points.jsonl or rank shards).",
    )
    parser.add_argument(
        "--bias-abs-threshold",
        type=float,
        default=0.0,
        help="Recall denominator/numerator only counts bias>0 with |bias| > this (default 0).",
    )
    parser.add_argument(
        "--q-cutoffs",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.5],
        help="Low-tail cutoffs; condition is q <= cutoff.",
    )
    parser.add_argument(
        "--recompute-q",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute length-bucket norm and q on loaded union (recommended for rank*.jsonl shards).",
    )
    parser.add_argument(
        "--suffix-len-bin-width",
        type=int,
        default=3,
        help="Must match collection if --recompute-q.",
    )
    parser.add_argument(
        "--suffix-len-min-count",
        type=int,
        default=20,
        help="Must match collection if --recompute-q.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional path to write full result JSON.",
    )
    args = parser.parse_args()

    paths = _expand_inputs(list(args.jsonl))
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    records = _load_jsonl_paths(paths)
    if args.recompute_q:
        _length_bucket_normalize(
            records,
            bin_width=int(args.suffix_len_bin_width),
            min_count=int(args.suffix_len_min_count),
        )
        _assign_q(records, key="f_suffix_rate_norm")

    tau = float(args.bias_abs_threshold)
    cutoffs = [float(x) for x in args.q_cutoffs]

    # Strong positive-bias set for recall (global, same for all cutoffs)
    strong_pos_mask = []
    for r in records:
        b = float(r.get("bias", float("nan")))
        strong_pos_mask.append(np.isfinite(b) and b > 0.0 and abs(b) > tau)
    strong_pos_idx = np.nonzero(np.asarray(strong_pos_mask))[0]
    n_strong = int(strong_pos_idx.size)

    rows_out: list[dict[str, Any]] = []
    print(f"n_records={len(records)}  n_strong_pos(|bias|>{tau}, bias>0)={n_strong}")
    print(
        "q_cutoff\tn_tail\tbias_pos_in_tail\tP(bias>0|q<=t)\t"
        f"recall_strong_pos_in_tail\tstrong_in_tail\tstrong_total"
    )
    for t in sorted(cutoffs):
        tail_idx = [
            i
            for i, r in enumerate(records)
            if np.isfinite(float(r.get("q", float("nan")))) and float(r["q"]) <= t
        ]
        n_tail = len(tail_idx)
        bias_pos_in_tail = 0
        for i in tail_idx:
            b = float(records[i].get("bias", float("nan")))
            if np.isfinite(b) and b > 0.0:
                bias_pos_in_tail += 1
        p_acc = (bias_pos_in_tail / n_tail) if n_tail > 0 else float("nan")

        strong_in_tail = sum(1 for i in tail_idx if strong_pos_mask[i])
        recall = (strong_in_tail / n_strong) if n_strong > 0 else float("nan")

        row = {
            "q_cutoff": t,
            "n_tail": n_tail,
            "bias_pos_in_tail": bias_pos_in_tail,
            "p_bias_pos_given_q_le_cutoff": p_acc,
            "n_strong_pos_total": n_strong,
            "n_strong_pos_in_tail": strong_in_tail,
            "recall_strong_pos_in_tail": recall,
            "bias_abs_threshold": tau,
        }
        rows_out.append(row)
        print(
            f"{t:g}\t{n_tail}\t{bias_pos_in_tail}\t{p_acc:.6g}\t{recall:.6g}\t{strong_in_tail}\t{n_strong}"
        )

    payload = {
        "inputs": [str(p) for p in paths],
        "recompute_q": bool(args.recompute_q),
        "suffix_len_bin_width": int(args.suffix_len_bin_width),
        "suffix_len_min_count": int(args.suffix_len_min_count),
        "bias_abs_threshold": tau,
        "rows": rows_out,
    }
    if args.out_json:
        outp = Path(args.out_json).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
