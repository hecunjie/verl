#!/usr/bin/env python3
"""从 sign_compare_merged.jsonl 统计「符号一致」在两种过滤下的指标。

约定（与 compare_bias_sign_bucket_vs_mc.py 的 trace 字段一致）：
- 「准确率」侧：只保留 f_bar_lookahead_2step >= threshold 的 step，再算 match 比例。
- 「召回」侧：只保留 f_bar_mc_128 >= threshold 的 step，再算 match 比例。

说明：这里的「准确率 / 召回」按你给的过滤条件分别定义在**不同子集**上的 match rate；
二者分母不同，不是同一套二分类的 precision/recall 分解，请在论文里写清定义。

match 默认使用 sign_match_real_trend_same_label（与 summary 中 trend_same_label 一致）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _iter_steps(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            trace = rec.get("trace_sign_compare")
            if not trace:
                continue
            for st in trace:
                if isinstance(st, dict):
                    out.append(st)
    return out


def _get_match(st: dict[str, Any], use_trend_all: bool) -> bool:
    key = "sign_match_real_trend_all" if use_trend_all else "sign_match_real_trend_same_label"
    return bool(st.get(key))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="sign_compare_merged.jsonl（每行一条 prompt 记录，含 trace_sign_compare）",
    )
    p.add_argument("--threshold", type=float, default=0.1, help="过滤阈值（默认 0.1）")
    p.add_argument(
        "--use_trend_all",
        action="store_true",
        help="用 sign_match_real_trend_all 代替 sign_match_real_trend_same_label",
    )
    args = p.parse_args()

    steps = _iter_steps(args.input)
    thr = float(args.threshold)
    use_all = bool(args.use_trend_all)

    # 准确率：过滤 f_bar_lookahead_2step < threshold
    acc_kept: list[dict[str, Any]] = []
    for st in steps:
        v = st.get("f_bar_lookahead_2step")
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv >= thr:
            acc_kept.append(st)

    n_acc = len(acc_kept)
    n_acc_match = sum(1 for st in acc_kept if _get_match(st, use_all))
    acc_rate = float(n_acc_match) / float(n_acc) if n_acc > 0 else float("nan")

    # 召回：过滤 f_bar_mc_128 < threshold
    rec_kept: list[dict[str, Any]] = []
    for st in steps:
        v = st.get("f_bar_mc_128")
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv >= thr:
            rec_kept.append(st)

    n_rec = len(rec_kept)
    n_rec_match = sum(1 for st in rec_kept if _get_match(st, use_all))
    rec_rate = float(n_rec_match) / float(n_rec) if n_rec > 0 else float("nan")

    # 交集：同一步同时满足两种过滤（便于对照）
    both: list[dict[str, Any]] = []
    for st in steps:
        la = st.get("f_bar_lookahead_2step")
        mc = st.get("f_bar_mc_128")
        if la is None or mc is None:
            continue
        try:
            if float(la) >= thr and float(mc) >= thr:
                both.append(st)
        except (TypeError, ValueError):
            continue
    n_both = len(both)
    n_both_match = sum(1 for st in both if _get_match(st, use_all))
    both_rate = float(n_both_match) / float(n_both) if n_both > 0 else float("nan")

    report = {
        "input": str(args.input.expanduser().resolve()),
        "threshold": thr,
        "use_trend_all": use_all,
        "num_steps_total_in_traces": len(steps),
        "accuracy_side": {
            "description": "keep steps with f_bar_lookahead_2step >= threshold; then match rate",
            "n_kept": n_acc,
            "n_match": n_acc_match,
            "rate": acc_rate,
        },
        "recall_side": {
            "description": "keep steps with f_bar_mc_128 >= threshold; then match rate",
            "n_kept": n_rec,
            "n_match": n_rec_match,
            "rate": rec_rate,
        },
        "intersection_both_filters": {
            "n_kept": n_both,
            "n_match": n_both_match,
            "rate": both_rate,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
