#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else float("nan")


def _bootstrap_ci_mean_diff(
    diffs: list[float],
    n_boot: int,
    ci_alpha: float,
    seed: int,
) -> dict[str, float]:
    if not diffs:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    n = len(diffs)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(float(np.mean(np.array(sample, dtype=np.float64))))
    lo_q = (1.0 - ci_alpha) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "mean_diff": float(np.mean(np.array(diffs, dtype=np.float64))),
        "ci_low": float(np.quantile(np.array(boots, dtype=np.float64), lo_q)),
        "ci_high": float(np.quantile(np.array(boots, dtype=np.float64), hi_q)),
    }


def _binom_two_sided_pvalue(n_trials: int, n_success: int) -> float:
    """Exact two-sided binomial test p-value for H0: p=0.5."""
    if n_trials <= 0:
        return float("nan")
    p_eq = 0.5 ** n_trials
    probs = [math.comb(n_trials, k) * p_eq for k in range(n_trials + 1)]
    p_obs = probs[n_success]
    p_two = 0.0
    for p in probs:
        if p <= p_obs + 1e-15:
            p_two += p
    return float(min(1.0, max(0.0, p_two)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Significance analysis for min-F_mc vs random-topk paired results.")
    parser.add_argument("--input", type=str, required=True, help="infer_compare_merged.jsonl path")
    parser.add_argument("--output", type=str, default="", help="output json path (default: same dir/significance.json)")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--ci_alpha", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap_samples should be >= 100.")
    if not (0.0 < float(args.ci_alpha) < 1.0):
        raise SystemExit("--ci_alpha must be in (0,1).")

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output.strip()
        else in_path.parent / "infer_compare_significance.json"
    )

    rows = _load_jsonl(in_path)
    if not rows:
        raise SystemExit("Empty input jsonl.")

    minf_ok: list[float] = []
    rand_ok: list[float] = []
    diffs: list[float] = []
    improve = 0
    regress = 0
    ties = 0
    for r in rows:
        m = 1.0 if bool((r.get("result_min_f_mc") or {}).get("is_correct", False)) else 0.0
        q = 1.0 if bool((r.get("result_random_topk") or {}).get("is_correct", False)) else 0.0
        minf_ok.append(m)
        rand_ok.append(q)
        diffs.append(m - q)
        if m > q:
            improve += 1
        elif m < q:
            regress += 1
        else:
            ties += 1

    n = len(rows)
    n_discordant = improve + regress
    acc_minf = _mean(minf_ok)
    acc_rand = _mean(rand_ok)
    gain = acc_minf - acc_rand
    rel_gain = (gain / acc_rand) if acc_rand > 0 else float("nan")

    boot = _bootstrap_ci_mean_diff(
        diffs=diffs,
        n_boot=int(args.bootstrap_samples),
        ci_alpha=float(args.ci_alpha),
        seed=int(args.seed),
    )

    # McNemar exact (binomial on discordant pairs)
    # H0: improve and regress equally likely (p=0.5)
    mcnemar_p = _binom_two_sided_pvalue(n_trials=n_discordant, n_success=improve)

    result = {
        "num_prompts": int(n),
        "accuracy_min_f_mc": float(acc_minf),
        "accuracy_random_topk": float(acc_rand),
        "accuracy_gain_abs": float(gain),
        "accuracy_gain_relative": float(rel_gain),
        "paired_counts": {
            "improve": int(improve),
            "regress": int(regress),
            "tie": int(ties),
            "discordant_total": int(n_discordant),
        },
        "bootstrap_mean_diff_ci": {
            "confidence": float(args.ci_alpha),
            "samples": int(args.bootstrap_samples),
            "mean_diff": float(boot["mean_diff"]),
            "ci_low": float(boot["ci_low"]),
            "ci_high": float(boot["ci_high"]),
        },
        "mcnemar_exact": {
            "p_value_two_sided": float(mcnemar_p),
            "note": "Exact binomial test on discordant pairs under H0 p=0.5.",
        },
        "input": str(in_path),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
