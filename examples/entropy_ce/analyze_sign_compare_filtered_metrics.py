#!/usr/bin/env python3
"""从 sign_compare_merged.jsonl 统计「符号一致」在多种过滤下的指标。

约定（与 compare_bias_sign_bucket_vs_mc.py 的 trace 字段一致）：
- 「准确率」侧（两种 lookahead 并行统计）：
  - lookahead_1step：f_bar_lookahead_1step >= threshold，且去掉
    |f_bar_lookahead_1step - f_real_lookahead_1step| < margin_threshold；
  - lookahead_2step：f_bar_lookahead_2step >= threshold，且去掉
    |f_bar_lookahead_2step - f_real_lookahead_2step| < margin_threshold。
- 「召回」侧（MC 参考，即 mc_m_samples_ref，字段名可能仍为 f_bar_mc_128）：
  保留 f_bar_mc_ref >= threshold 且 |f_bar_mc_ref - f_real_mc_ref| >= margin 的 step，
  match 用 trend 与 **MC ref** 的符号是否一致（sign_match_real_trend_same_label）。
- 「召回」侧（MC compare，即 mc_m_samples_compare，如 M=1）：
  保留 f_bar_mc_compare / f_real_mc_compare 满足同样阈值与 margin，
  match 用 trend 与 **MC compare** 的符号是否一致（或由 sign_mc_compare 与 trend 现场比对）。

说明：准确率 / 召回 定义在**不同子集**上的 match rate；分母不同。
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


def _acc_step_ok_1step(st: dict[str, Any], thr: float, margin_thr: float) -> bool:
    """准确率子集（1-step）：f_bar_la1>=thr 且 |f_bar-f_real|>=margin。"""
    fb = st.get("f_bar_lookahead_1step")
    fr = st.get("f_real_lookahead_1step")
    if fb is None or fr is None:
        return False
    try:
        fbf, frf = float(fb), float(fr)
    except (TypeError, ValueError):
        return False
    if fbf < thr:
        return False
    if abs(fbf - frf) < margin_thr:
        return False
    return True


def _acc_step_ok_2step(st: dict[str, Any], thr: float, margin_thr: float) -> bool:
    """准确率子集（2-step）：f_bar_la2>=thr 且 |f_bar-f_real|>=margin。"""
    fb = st.get("f_bar_lookahead_2step")
    fr = st.get("f_real_lookahead_2step")
    if fb is None or fr is None:
        return False
    try:
        fbf, frf = float(fb), float(fr)
    except (TypeError, ValueError):
        return False
    if fbf < thr:
        return False
    if abs(fbf - frf) < margin_thr:
        return False
    return True


def _rec_step_ok_ref(st: dict[str, Any], thr: float, margin_thr: float) -> bool:
    """召回子集（MC ref）：f_bar_mc_ref 或旧名 f_bar_mc_128。"""
    fb = st.get("f_bar_mc_ref", st.get("f_bar_mc_128"))
    fr = st.get("f_real_mc_ref", st.get("f_real_mc_128"))
    if fb is None or fr is None:
        return False
    try:
        fbf, frf = float(fb), float(fr)
    except (TypeError, ValueError):
        return False
    if fbf < thr:
        return False
    if abs(fbf - frf) < margin_thr:
        return False
    return True


def _rec_step_ok_compare(st: dict[str, Any], thr: float, margin_thr: float) -> bool:
    """召回子集（MC compare，需跑过 mc_m_samples_compare>0）。"""
    fb = st.get("f_bar_mc_compare")
    fr = st.get("f_real_mc_compare")
    if fb is None or fr is None:
        return False
    try:
        fbf, frf = float(fb), float(fr)
    except (TypeError, ValueError):
        return False
    if fbf < thr:
        return False
    if abs(fbf - frf) < margin_thr:
        return False
    return True


def _get_match_vs_mc_compare(st: dict[str, Any], use_trend_all: bool) -> bool:
    """trend 符号是否与 MC compare 的 sign(f_bar_mc - f_real_mc) 一致。"""
    sc = st.get("sign_mc_compare")
    if sc is None:
        return False
    key = "sign_real_trend_all" if use_trend_all else "sign_real_trend_same_label"
    st_tr = st.get(key)
    if st_tr is None:
        return False
    return int(st_tr) == int(sc)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="sign_compare_merged.jsonl（每行一条 prompt 记录，含 trace_sign_compare）",
    )
    p.add_argument("--threshold", type=float, default=0.1, help="f_bar 下限过滤阈值（默认 0.1）")
    p.add_argument(
        "--margin_threshold",
        type=float,
        default=0.1,
        help="|f_bar - f_real| 下限：小于此值的 step 会被排除（默认 0.1）",
    )
    p.add_argument(
        "--use_trend_all",
        action="store_true",
        help="用 sign_match_real_trend_all 代替 sign_match_real_trend_same_label",
    )
    args = p.parse_args()

    steps = _iter_steps(args.input)
    thr = float(args.threshold)
    margin_thr = float(args.margin_threshold)
    use_all = bool(args.use_trend_all)

    # 准确率（1-step）
    acc1_kept = [st for st in steps if _acc_step_ok_1step(st, thr, margin_thr)]
    n_acc1 = len(acc1_kept)
    n_acc1_match = sum(1 for st in acc1_kept if _get_match(st, use_all))
    acc1_rate = float(n_acc1_match) / float(n_acc1) if n_acc1 > 0 else float("nan")

    # 准确率（2-step）
    acc_kept = [st for st in steps if _acc_step_ok_2step(st, thr, margin_thr)]

    n_acc = len(acc_kept)
    n_acc_match = sum(1 for st in acc_kept if _get_match(st, use_all))
    acc_rate = float(n_acc_match) / float(n_acc) if n_acc > 0 else float("nan")

    # 召回（MC ref）
    rec_ref_kept = [st for st in steps if _rec_step_ok_ref(st, thr, margin_thr)]
    n_rec_ref = len(rec_ref_kept)
    n_rec_ref_match = sum(1 for st in rec_ref_kept if _get_match(st, use_all))
    rec_ref_rate = float(n_rec_ref_match) / float(n_rec_ref) if n_rec_ref > 0 else float("nan")

    # 召回（MC compare，如 M=1）
    rec_cmp_kept = [st for st in steps if _rec_step_ok_compare(st, thr, margin_thr)]
    n_rec_cmp = len(rec_cmp_kept)
    n_rec_cmp_match = sum(1 for st in rec_cmp_kept if _get_match_vs_mc_compare(st, use_all))
    rec_cmp_rate = float(n_rec_cmp_match) / float(n_rec_cmp) if n_rec_cmp > 0 else float("nan")
    # 同一子集上：trend 与 MC ref 符号是否一致（与 recall_mc_ref 的 match 定义对齐，便于对照）
    n_rec_cmp_match_vs_ref = sum(1 for st in rec_cmp_kept if _get_match(st, use_all))
    rec_cmp_rate_vs_ref_sign = (
        float(n_rec_cmp_match_vs_ref) / float(n_rec_cmp) if n_rec_cmp > 0 else float("nan")
    )

    # 交集：lookahead_1step 准确率 ∧ MC ref 召回
    both1 = [
        st for st in steps if _acc_step_ok_1step(st, thr, margin_thr) and _rec_step_ok_ref(st, thr, margin_thr)
    ]
    n_both1 = len(both1)
    n_both1_match = sum(1 for st in both1 if _get_match(st, use_all))
    both1_rate = float(n_both1_match) / float(n_both1) if n_both1 > 0 else float("nan")

    # 交集：lookahead_1step ∧ MC compare 召回
    both1_cmp = [
        st
        for st in steps
        if _acc_step_ok_1step(st, thr, margin_thr) and _rec_step_ok_compare(st, thr, margin_thr)
    ]
    n_both1_cmp = len(both1_cmp)
    n_both1_cmp_match = sum(1 for st in both1_cmp if _get_match_vs_mc_compare(st, use_all))
    both1_cmp_rate = float(n_both1_cmp_match) / float(n_both1_cmp) if n_both1_cmp > 0 else float("nan")
    n_both1_cmp_match_vs_ref = sum(1 for st in both1_cmp if _get_match(st, use_all))
    both1_cmp_rate_vs_ref = (
        float(n_both1_cmp_match_vs_ref) / float(n_both1_cmp) if n_both1_cmp > 0 else float("nan")
    )

    # 交集：lookahead_2step 准确率 ∧ MC ref 召回
    both = [
        st for st in steps if _acc_step_ok_2step(st, thr, margin_thr) and _rec_step_ok_ref(st, thr, margin_thr)
    ]
    n_both = len(both)
    n_both_match = sum(1 for st in both if _get_match(st, use_all))
    both_rate = float(n_both_match) / float(n_both) if n_both > 0 else float("nan")

    # 交集：lookahead_2step ∧ MC compare 召回
    both2_cmp = [
        st
        for st in steps
        if _acc_step_ok_2step(st, thr, margin_thr) and _rec_step_ok_compare(st, thr, margin_thr)
    ]
    n_both2_cmp = len(both2_cmp)
    n_both2_cmp_match = sum(1 for st in both2_cmp if _get_match_vs_mc_compare(st, use_all))
    both2_cmp_rate = float(n_both2_cmp_match) / float(n_both2_cmp) if n_both2_cmp > 0 else float("nan")
    n_both2_cmp_match_vs_ref = sum(1 for st in both2_cmp if _get_match(st, use_all))
    both2_cmp_rate_vs_ref = (
        float(n_both2_cmp_match_vs_ref) / float(n_both2_cmp) if n_both2_cmp > 0 else float("nan")
    )

    report = {
        "input": str(args.input.expanduser().resolve()),
        "threshold": thr,
        "margin_threshold": margin_thr,
        "use_trend_all": use_all,
        "num_steps_total_in_traces": len(steps),
        "accuracy_side_lookahead_1step": {
            "description": (
                "keep steps with f_bar_lookahead_1step >= threshold AND "
                "|f_bar_lookahead_1step - f_real_lookahead_1step| >= margin_threshold; then match rate"
            ),
            "n_kept": n_acc1,
            "n_match": n_acc1_match,
            "rate": acc1_rate,
        },
        "accuracy_side_lookahead_2step": {
            "description": (
                "keep steps with f_bar_lookahead_2step >= threshold AND "
                "|f_bar_lookahead_2step - f_real_lookahead_2step| >= margin_threshold; then match rate"
            ),
            "n_kept": n_acc,
            "n_match": n_acc_match,
            "rate": acc_rate,
        },
        "accuracy_side": {
            "description": (
                "alias of accuracy_side_lookahead_2step for backward compatibility "
                "(same as f_bar_lookahead_2step branch)"
            ),
            "n_kept": n_acc,
            "n_match": n_acc_match,
            "rate": acc_rate,
        },
        "recall_side_mc_ref": {
            "description": (
                "keep steps with f_bar_mc_ref (or f_bar_mc_128) >= threshold AND "
                "|f_bar - f_real| >= margin; match = trend vs MC ref sign (same as sign_match_real_trend_same_label)"
            ),
            "n_kept": n_rec_ref,
            "n_match": n_rec_ref_match,
            "rate": rec_ref_rate,
        },
        "recall_side_mc_compare": {
            "description": (
                "keep steps with f_bar_mc_compare >= threshold AND "
                "|f_bar_mc_compare - f_real_mc_compare| >= margin; "
                "match = trend vs MC compare sign (requires mc_m_samples_compare>0 in run)"
            ),
            "n_kept": n_rec_cmp,
            "n_match": n_rec_cmp_match,
            "rate": rec_cmp_rate,
            "n_match_trend_vs_mc_ref_sign": n_rec_cmp_match_vs_ref,
            "rate_trend_vs_mc_ref_sign_on_same_subset": rec_cmp_rate_vs_ref_sign,
            "note": (
                "rate uses trend vs sign_mc_compare; rate_trend_vs_mc_ref_sign_on_same_subset uses "
                "the same match as recall_side_mc_ref (trend vs MC ref) but only on MC-compare filter."
            ),
        },
        "recall_side": {
            "description": "alias of recall_side_mc_ref (backward compatibility)",
            "n_kept": n_rec_ref,
            "n_match": n_rec_ref_match,
            "rate": rec_ref_rate,
        },
        "intersection_both_filters_lookahead_1step": {
            "description": "1step accuracy filters AND recall_side_mc_ref",
            "n_kept": n_both1,
            "n_match": n_both1_match,
            "rate": both1_rate,
        },
        "intersection_both_filters_lookahead_1step_mc_compare": {
            "description": "1step accuracy filters AND recall_side_mc_compare",
            "n_kept": n_both1_cmp,
            "n_match_trend_vs_mc_compare": n_both1_cmp_match,
            "rate_trend_vs_mc_compare": both1_cmp_rate,
            "n_match_trend_vs_mc_ref": n_both1_cmp_match_vs_ref,
            "rate_trend_vs_mc_ref": both1_cmp_rate_vs_ref,
        },
        "intersection_both_filters_lookahead_2step": {
            "description": "2step accuracy filters AND recall_side_mc_ref",
            "n_kept": n_both,
            "n_match": n_both_match,
            "rate": both_rate,
        },
        "intersection_both_filters_lookahead_2step_mc_compare": {
            "description": "2step accuracy filters AND recall_side_mc_compare (parallel to lookahead2step+mc128 setup)",
            "n_kept": n_both2_cmp,
            "n_match_trend_vs_mc_compare": n_both2_cmp_match,
            "rate_trend_vs_mc_compare": both2_cmp_rate,
            "n_match_trend_vs_mc_ref": n_both2_cmp_match_vs_ref,
            "rate_trend_vs_mc_ref": both2_cmp_rate_vs_ref,
        },
        "intersection_both_filters": {
            "description": (
                "alias of intersection_both_filters_lookahead_2step (2step + mc ref) for backward compatibility"
            ),
            "n_kept": n_both,
            "n_match": n_both_match,
            "rate": both_rate,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
