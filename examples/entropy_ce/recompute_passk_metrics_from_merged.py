#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else float("nan")


def _pass_at_k_from_n_c(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n <= 0:
        return float("nan")
    kk = min(max(int(k), 1), int(n))
    cc = min(max(int(c), 0), int(n))
    if n - cc < kk:
        return 1.0
    prod = 1.0
    for i in range(kk):
        prod *= float(n - cc - i) / float(n - i)
    return 1.0 - prod


def _extract_sample_correct(rec: dict[str, Any]) -> list[float]:
    vals = rec.get("sample_correct")
    if not isinstance(vals, list):
        raise ValueError("record missing 'sample_correct' list")
    out: list[float] = []
    for x in vals:
        if isinstance(x, bool):
            out.append(1.0 if x else 0.0)
        else:
            out.append(1.0 if float(x) > 0.5 else 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute pass@k and mean@n from merged.jsonl.")
    parser.add_argument("--input", type=str, required=True, help="Path to merged.jsonl (must include sample_correct).")
    parser.add_argument("--k_small", type=int, default=4)
    parser.add_argument("--k_large", type=int, default=32)
    parser.add_argument(
        "--mean_n",
        type=int,
        default=32,
        help="Compute mean@n per prompt using first n samples (clipped to available length).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output json path. Default: <input_dir>/recomputed_passk_summary.json",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    rows: list[dict[str, Any]] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    if not rows:
        raise SystemExit("input merged.jsonl is empty")

    pass_small_vals: list[float] = []
    pass_large_vals: list[float] = []
    mean_n_vals: list[float] = []
    n_samples_vals: list[int] = []

    for rec in rows:
        corr = _extract_sample_correct(rec)
        n = len(corr)
        c = int(sum(1 for x in corr if x > 0.5))
        n_samples_vals.append(n)

        pass_small_vals.append(_pass_at_k_from_n_c(n=n, c=c, k=int(args.k_small)))
        pass_large_vals.append(_pass_at_k_from_n_c(n=n, c=c, k=int(args.k_large)))

        m = min(max(int(args.mean_n), 1), n)
        mean_n_vals.append(float(sum(corr[:m])) / float(m))

    summary = {
        "num_prompts": int(len(rows)),
        f"pass@{int(args.k_small)}": _mean(pass_small_vals),
        f"pass@{int(args.k_large)}": _mean(pass_large_vals),
        f"mean@{int(args.mean_n)}": _mean(mean_n_vals),
        "avg_num_samples_per_prompt": _mean([float(x) for x in n_samples_vals]),
        "source_file": str(in_path),
        "estimator": "1 - C(n-c,k)/C(n,k)",
    }

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else in_path.parent / "recomputed_passk_summary.json"
    )
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] summary -> {out_path}")


if __name__ == "__main__":
    main()

