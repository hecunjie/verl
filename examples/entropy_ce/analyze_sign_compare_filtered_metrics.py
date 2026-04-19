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

**sign_match_mc_compare_vs_ref 的 precision / recall**（见
``precision_recall_sign_match_mc_compare_vs_ref``）：
- predict = mc_compare 符号，gt = mc_ref 符号，Y = ``sign_match_mc_compare_vs_ref``。
- **仅使用你在命令行显式传入的条件**（``--pr_precision_*`` / ``--pr_recall_*``）；
  未传的项不参与过滤。若某一侧三个条件都未传，则该侧分母为「``sign_match_mc_compare_vs_ref``
  非空」的全部 step。
- 可选条件（均可单独或组合出现）：``min_f_bar``、``min_abs_gap``（|f_bar−f_real| 下限）、
  ``relative_gap_frac``（|f_bar−f_real| ≥ frac×|f_bar|，且 frac>0 时才生效）。

报告中的 ``value_stratification_by_real_rollout_is_correct``：在 precision / recall 分母子集上，按
**该条 prompt 的** ``real_rollout_is_correct``（整题回答是否正确）分组，分别输出 compare / ref 侧
f_bar、f_real、|Δf|、相对差分、entropy_t 等的计数与均值（与逐步 ``sign_match_mc_compare_vs_ref`` 无关）。
"""

from __future__ import annotations

import argparse
import json
import math
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
            parent_rc = rec.get("real_rollout_is_correct")
            for st in trace:
                if isinstance(st, dict):
                    # 来自 prompt 级字段，供按「整题是否正确」分层统计
                    out.append({**st, "real_rollout_is_correct": parent_rc})
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


def _relative_gap_ok(f_bar: Any, f_real: Any, rel_frac: float, *, eps: float = 1e-12) -> bool:
    """|f_bar - f_real| >= rel_frac * |f_bar|（rel_frac>0）。"""
    if rel_frac <= 0:
        return True
    if f_bar is None or f_real is None:
        return False
    try:
        fb = float(f_bar)
        fr = float(f_real)
    except (TypeError, ValueError):
        return False
    afb = abs(fb)
    if afb < eps:
        return False
    return abs(fb - fr) >= rel_frac * afb


def _passes_optional_fb_fr_filters(
    fb: Any,
    fr: Any,
    min_fbar: float | None,
    min_abs_gap: float | None,
    relative_gap_frac: float | None,
) -> bool:
    """未指定的条件不参与。若至少指定一项，则要求 fb/fr 存在且可解析。"""
    if min_fbar is None and min_abs_gap is None and relative_gap_frac is None:
        return True
    if fb is None or fr is None:
        return False
    try:
        fbf, frf = float(fb), float(fr)
    except (TypeError, ValueError):
        return False
    if min_fbar is not None and fbf < float(min_fbar):
        return False
    if min_abs_gap is not None and abs(fbf - frf) < float(min_abs_gap):
        return False
    if relative_gap_frac is not None and float(relative_gap_frac) > 0:
        if not _relative_gap_ok(fbf, frf, float(relative_gap_frac)):
            return False
    return True


def _include_precision_pr_subset(
    st: dict[str, Any],
    min_fbar: float | None,
    min_abs_gap: float | None,
    relative_gap_frac: float | None,
) -> bool:
    if _get_sign_match_mc_compare_vs_ref(st) is None:
        return False
    if min_fbar is None and min_abs_gap is None and relative_gap_frac is None:
        return True
    return _passes_optional_fb_fr_filters(
        st.get("f_bar_mc_compare"),
        st.get("f_real_mc_compare"),
        min_fbar,
        min_abs_gap,
        relative_gap_frac,
    )


def _include_recall_pr_subset(
    st: dict[str, Any],
    min_fbar: float | None,
    min_abs_gap: float | None,
    relative_gap_frac: float | None,
) -> bool:
    if _get_sign_match_mc_compare_vs_ref(st) is None:
        return False
    if min_fbar is None and min_abs_gap is None and relative_gap_frac is None:
        return True
    fb = st.get("f_bar_mc_ref", st.get("f_bar_mc_128"))
    fr = st.get("f_real_mc_ref", st.get("f_real_mc_128"))
    return _passes_optional_fb_fr_filters(fb, fr, min_fbar, min_abs_gap, relative_gap_frac)


def _get_sign_match_mc_compare_vs_ref(st: dict[str, Any]) -> bool | None:
    """None = 未计算（未跑 compare 或字段缺失）。"""
    if "sign_match_mc_compare_vs_ref" not in st:
        return None
    v = st.get("sign_match_mc_compare_vs_ref")
    if v is None:
        return None
    return bool(v)


def _mean_finite(xs: list[float]) -> float:
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _summarize_compare_trace_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """mc_compare 侧标量：用于 precision 分母内正确/错误子集。"""
    fbs: list[float] = []
    frs: list[float] = []
    gaps: list[float] = []
    rels: list[float] = []
    ents: list[float] = []
    for st in steps:
        fb, fr = st.get("f_bar_mc_compare"), st.get("f_real_mc_compare")
        if fb is None or fr is None:
            continue
        try:
            fbf, frf = float(fb), float(fr)
        except (TypeError, ValueError):
            continue
        fbs.append(fbf)
        frs.append(frf)
        g = abs(fbf - frf)
        gaps.append(g)
        afb = abs(fbf)
        rels.append(g / afb if afb >= 1e-12 else float("nan"))
        et = st.get("entropy_t")
        if et is not None:
            try:
                ents.append(float(et))
            except (TypeError, ValueError):
                pass
    return {
        "n_steps_in_group": len(steps),
        "n_steps_with_compare_f_numeric": len(fbs),
        "mean_f_bar_mc_compare": _mean_finite(fbs),
        "mean_f_real_mc_compare": _mean_finite(frs),
        "mean_abs_gap": _mean_finite(gaps),
        "mean_relative_gap_ratio": _mean_finite(rels),
        "mean_entropy_t": _mean_finite(ents),
    }


def _summarize_ref_trace_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """mc_ref 侧标量：用于 recall 分母内正确/错误子集。"""
    fbs: list[float] = []
    frs: list[float] = []
    gaps: list[float] = []
    rels: list[float] = []
    ents: list[float] = []
    for st in steps:
        fb = st.get("f_bar_mc_ref", st.get("f_bar_mc_128"))
        fr = st.get("f_real_mc_ref", st.get("f_real_mc_128"))
        if fb is None or fr is None:
            continue
        try:
            fbf, frf = float(fb), float(fr)
        except (TypeError, ValueError):
            continue
        fbs.append(fbf)
        frs.append(frf)
        g = abs(fbf - frf)
        gaps.append(g)
        afb = abs(fbf)
        rels.append(g / afb if afb >= 1e-12 else float("nan"))
        et = st.get("entropy_t")
        if et is not None:
            try:
                ents.append(float(et))
            except (TypeError, ValueError):
                pass
    return {
        "n_steps_in_group": len(steps),
        "n_steps_with_ref_f_numeric": len(fbs),
        "mean_f_bar_mc_ref": _mean_finite(fbs),
        "mean_f_real_mc_ref": _mean_finite(frs),
        "mean_abs_gap": _mean_finite(gaps),
        "mean_relative_gap_ratio": _mean_finite(rels),
        "mean_entropy_t": _mean_finite(ents),
    }


def _build_value_stratification_by_real_rollout(
    prec_pr_steps: list[dict[str, Any]],
    rec_pr_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """按 prompt 级 ``real_rollout_is_correct`` 分答对/答错/未知，分别汇总 compare / ref 侧数值。"""

    def _split_rollout(
        steps: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        corr = [st for st in steps if st.get("real_rollout_is_correct") is True]
        wrong = [st for st in steps if st.get("real_rollout_is_correct") is False]
        unk = [st for st in steps if st.get("real_rollout_is_correct") is None]
        return corr, wrong, unk

    p_corr, p_wrong, p_unk = _split_rollout(prec_pr_steps)
    r_corr, r_wrong, r_unk = _split_rollout(rec_pr_steps)
    return {
        "description": (
            "Within PR precision/recall denominators, split by parent prompt field real_rollout_is_correct "
            "(whole-answer correctness), not by sign_match_mc_compare_vs_ref. "
            "Compare fields: f_bar_mc_compare; ref fields: f_bar_mc_ref. "
            "unknown = field missing or null on merged jsonl row."
        ),
        "precision_denominator": {
            "real_rollout_is_correct_true": {
                "count": len(p_corr),
                "compare_side_fields": _summarize_compare_trace_steps(p_corr),
            },
            "real_rollout_is_correct_false": {
                "count": len(p_wrong),
                "compare_side_fields": _summarize_compare_trace_steps(p_wrong),
            },
            "real_rollout_is_correct_unknown": {
                "count": len(p_unk),
                "compare_side_fields": _summarize_compare_trace_steps(p_unk),
            },
        },
        "recall_denominator": {
            "real_rollout_is_correct_true": {
                "count": len(r_corr),
                "ref_side_fields": _summarize_ref_trace_steps(r_corr),
            },
            "real_rollout_is_correct_false": {
                "count": len(r_wrong),
                "ref_side_fields": _summarize_ref_trace_steps(r_wrong),
            },
            "real_rollout_is_correct_unknown": {
                "count": len(r_unk),
                "ref_side_fields": _summarize_ref_trace_steps(r_unk),
            },
        },
    }


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
    p.add_argument(
        "--pr_precision_min_f_bar",
        type=float,
        default=None,
        help="PR precision 分母：仅当传入时要求 f_bar_mc_compare >= 该值",
    )
    p.add_argument(
        "--pr_precision_min_abs_gap",
        type=float,
        default=None,
        help="PR precision 分母：仅当传入时要求 |f_bar_mc_compare - f_real_mc_compare| >= 该值",
    )
    p.add_argument(
        "--pr_precision_relative_gap_frac",
        type=float,
        default=None,
        help="PR precision 分母：仅当传入且 >0 时要求 |Δf| >= frac*|f_bar|（compare 侧）",
    )
    p.add_argument(
        "--pr_recall_min_f_bar",
        type=float,
        default=None,
        help="PR recall 分母：仅当传入时要求 f_bar_mc_ref >= 该值",
    )
    p.add_argument(
        "--pr_recall_min_abs_gap",
        type=float,
        default=None,
        help="PR recall 分母：仅当传入时要求 |f_bar_mc_ref - f_real_mc_ref| >= 该值",
    )
    p.add_argument(
        "--pr_recall_relative_gap_frac",
        type=float,
        default=None,
        help="PR recall 分母：仅当传入且 >0 时要求 |Δf| >= frac*|f_bar|（ref 侧）",
    )
    args = p.parse_args()

    steps = _iter_steps(args.input)
    thr = float(args.threshold)
    margin_thr = float(args.margin_threshold)
    pp_f = args.pr_precision_min_f_bar
    pp_a = args.pr_precision_min_abs_gap
    pp_r = args.pr_precision_relative_gap_frac
    pr_f = args.pr_recall_min_f_bar
    pr_a = args.pr_recall_min_abs_gap
    pr_r = args.pr_recall_relative_gap_frac
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

    # --- Precision / Recall for sign_match_mc_compare_vs_ref (predict=compare, gt=ref) ---
    prec_pr_steps = [
        st for st in steps if _include_precision_pr_subset(st, pp_f, pp_a, pp_r)
    ]
    n_pr_prec = len(prec_pr_steps)
    n_pr_prec_pos = sum(1 for st in prec_pr_steps if _get_sign_match_mc_compare_vs_ref(st))
    pr_precision = float(n_pr_prec_pos) / float(n_pr_prec) if n_pr_prec > 0 else float("nan")

    rec_pr_steps = [st for st in steps if _include_recall_pr_subset(st, pr_f, pr_a, pr_r)]
    n_pr_rec = len(rec_pr_steps)
    n_pr_rec_pos = sum(1 for st in rec_pr_steps if _get_sign_match_mc_compare_vs_ref(st))
    pr_recall = float(n_pr_rec_pos) / float(n_pr_rec) if n_pr_rec > 0 else float("nan")

    if (
        math.isfinite(pr_precision)
        and math.isfinite(pr_recall)
        and (pr_precision + pr_recall) > 0
    ):
        pr_f1 = 2.0 * pr_precision * pr_recall / (pr_precision + pr_recall)
    else:
        pr_f1 = float("nan")

    value_stratification_by_real_rollout_is_correct = _build_value_stratification_by_real_rollout(
        prec_pr_steps, rec_pr_steps
    )

    report = {
        "input": str(args.input.expanduser().resolve()),
        "threshold": thr,
        "margin_threshold": margin_thr,
        "pr_filters": {
            "precision": {
                "min_f_bar": pp_f,
                "min_abs_gap": pp_a,
                "relative_gap_frac": pp_r,
            },
            "recall": {
                "min_f_bar": pr_f,
                "min_abs_gap": pr_a,
                "relative_gap_frac": pr_r,
            },
            "note": "仅非 null 的项参与过滤；某一侧若全为 null，则该侧使用全部 sign_match 已定义的 step。",
        },
        "use_trend_all": use_all,
        "num_steps_total_in_traces": len(steps),
        "precision_recall_sign_match_mc_compare_vs_ref": {
            "description": (
                "predict = sign(mc_compare), gt = sign(mc_ref), Y = sign_match_mc_compare_vs_ref. "
                "Only CLI args under pr_filters that are non-null apply; unspecified filters are ignored. "
                "If all precision filters are null, precision denominator = all steps with Y defined. "
                "If all recall filters are null, recall denominator = same. "
                "If any filter is set on a side, that side requires valid f_bar/f_real for the checks."
            ),
            "precision": {
                "denominator_steps": n_pr_prec,
                "numerator_positive": n_pr_prec_pos,
                "rate": pr_precision,
            },
            "recall": {
                "denominator_steps": n_pr_rec,
                "numerator_positive": n_pr_rec_pos,
                "rate": pr_recall,
            },
            "f1": pr_f1,
        },
        "value_stratification_by_real_rollout_is_correct": value_stratification_by_real_rollout_is_correct,
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
