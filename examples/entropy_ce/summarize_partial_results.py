#!/usr/bin/env python3
"""
Summarize partial entropy-credit experiment results from JSONL shards.

This script is designed for the situation where the main run crashed near the end
but some `phase1_3_rank*.jsonl` files were already written.

Inputs (searched in output_dir):
- phase1_3_merged.jsonl (preferred if exists)
- otherwise phase1_3_rank*.jsonl

Output:
- metrics_partial.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any

import numpy as np


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    xr = rankdata_average_ties(xa)
    yr = rankdata_average_ties(ya)
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    denom = math.sqrt(float((xc * xc).sum() * (yc * yc).sum()))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float((xc * yc).sum() / denom)


def precision_at_k(signal: list[float], importance: list[float], k: int) -> float:
    n = min(len(signal), len(importance))
    if n <= 0:
        return float("nan")
    k = min(k, n)
    sig_idx = np.argsort(np.asarray(signal))[-k:]
    imp_idx = set(np.argsort(np.asarray(importance))[-k:].tolist())
    hit = sum(int(i in imp_idx) for i in sig_idx.tolist())
    return float(hit / k)


def mean_no_nan(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def iter_records(files: list[str]):
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", ""),
        help="Experiment output directory containing phase1_3_*.jsonl.",
    )
    args = ap.parse_args()

    if not args.output_dir:
        raise SystemExit("Please pass --output_dir or set OUTPUT_DIR environment variable.")

    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    merged = os.path.join(out_dir, "phase1_3_merged.jsonl")
    if os.path.exists(merged):
        files = [merged]
    else:
        files = sorted(glob.glob(os.path.join(out_dir, "phase1_3_rank*.jsonl")))

    if not files:
        raise SystemExit(f"No phase1_3 files found in {out_dir}")

    corr_delta: list[float] = []
    corr_h: list[float] = []
    corr_v: list[float] = []
    p5_delta: list[float] = []
    p10_delta: list[float] = []
    p20_delta: list[float] = []
    p5_h: list[float] = []
    p10_h: list[float] = []
    p20_h: list[float] = []

    num_records = 0
    used = 0
    skipped_missing = 0

    for rec in iter_records(files):
        num_records += 1
        try:
            imp = rec.get("importance_method_b") or []
            if not any(float(x) > 0 for x in imp):
                continue
            delta = rec["deltas"]
            h = rec["entropies"]
            v = rec["varentropies"]
        except Exception:
            skipped_missing += 1
            continue

        used += 1
        corr_delta.append(spearman(delta, imp))
        corr_h.append(spearman(h, imp))
        corr_v.append(spearman(v, imp))

        p5_delta.append(precision_at_k(delta, imp, 5))
        p10_delta.append(precision_at_k(delta, imp, 10))
        p20_delta.append(precision_at_k(delta, imp, 20))
        p5_h.append(precision_at_k(h, imp, 5))
        p10_h.append(precision_at_k(h, imp, 10))
        p20_h.append(precision_at_k(h, imp, 20))

    metrics: dict[str, Any] = {
        "output_dir": out_dir,
        "input_files": files,
        "num_rollout_records": num_records,
        "num_rollouts_with_nonzero_importance": used,
        "skipped_due_to_missing_fields": skipped_missing,
        "spearman_delta_vs_importance": mean_no_nan(corr_delta),
        "spearman_entropy_vs_importance": mean_no_nan(corr_h),
        "spearman_varentropy_vs_importance": mean_no_nan(corr_v),
        "precision@5_delta": mean_no_nan(p5_delta),
        "precision@10_delta": mean_no_nan(p10_delta),
        "precision@20_delta": mean_no_nan(p20_delta),
        "precision@5_entropy": mean_no_nan(p5_h),
        "precision@10_entropy": mean_no_nan(p10_h),
        "precision@20_entropy": mean_no_nan(p20_h),
    }

    out_path = os.path.join(out_dir, "metrics_partial.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

