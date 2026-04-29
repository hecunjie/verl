#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    try:
        x = float(str(v))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return int(v)
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def _extract_step_from_filename(path: Path) -> int | None:
    stem = path.stem
    m = re.search(r"(\d+)", stem)
    if not m:
        return None
    return int(m.group(1))


def _iter_jsonl_paths(input_dir: Path) -> list[Path]:
    files = [p for p in input_dir.glob("*.jsonl") if p.is_file()]
    files.sort(key=lambda p: (_extract_step_from_filename(p) is None, _extract_step_from_filename(p) or 10**18, p.name))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "统计 FEPO 数据在每个 step 上的 "
            "P(correct|low-tail)/P(correct|high-head) 和 "
            "E[advantage|low-tail]/E[advantage|high-head] 并画图。"
        )
    )
    parser.add_argument("--input_dir", type=str, required=True, help="包含多个 step.jsonl 的目录。")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录。")
    parser.add_argument(
        "--group_mode",
        choices=["suffix_rate", "m"],
        default="m",
        help="low-tail/high-head 分组方式：suffix_rate=按 f_suffix_rate 切分；m=按 m>1 或 m<1。",
    )
    parser.add_argument(
        "--suffix_rate_key",
        type=str,
        default="f_suffix_rate",
        help="group_mode=suffix_rate 时用于切分的字段名。",
    )
    parser.add_argument(
        "--m_key",
        type=str,
        default="m",
        help="group_mode=m 时使用的权重字段名（m>1: low-tail, m<1: high-head）。",
    )
    parser.add_argument(
        "--advantage_key",
        type=str,
        default="adv_after",
        help="advantage 字段名（可改成 adv_before）。",
    )
    parser.add_argument(
        "--split_mode",
        choices=["median", "fixed"],
        default="median",
        help="low-tail/high-head 切分方式：median=每个 step 用中位数切分；fixed=全局固定阈值。",
    )
    parser.add_argument(
        "--fixed_suffix_threshold",
        type=float,
        default=0.2,
        help="split_mode=fixed 时使用的阈值，<= 阈值为 low-tail，> 阈值为 high-head。",
    )
    parser.add_argument(
        "--min_group_count",
        type=int,
        default=1,
        help="每个组在每个 step 的最小样本数，不足则该 step 记为 NaN。",
    )
    parser.add_argument(
        "--m_equal_eps",
        type=float,
        default=1e-8,
        help="group_mode=m 时，将 |m-1|<=eps 视为 m==1 并过滤。",
    )
    parser.add_argument(
        "--adv_zero_eps",
        type=float,
        default=1e-12,
        help="计算 P(correct) 时，将 |adv|<=eps 视为 0 并从分母剔除。",
    )
    parser.add_argument("--q_key", type=str, default="q", help="用于 q 近邻统计的字段名。")
    parser.add_argument("--h_t_key", type=str, default="h_t", help="用于 q 近邻统计的 token 熵字段名。")
    parser.add_argument(
        "--q_target_high",
        type=float,
        default=0.8,
        help="q 近邻统计中的高目标值。",
    )
    parser.add_argument(
        "--q_target_low",
        type=float,
        default=0.2,
        help="q 近邻统计中的低目标值。",
    )
    parser.add_argument(
        "--quantile_levels",
        type=str,
        default="0.1,0.2,0.3,0.5,0.8,0.9",
        help="用于分位图的分位点，逗号分隔。",
    )
    args = parser.parse_args()
    quantile_levels = [float(x.strip()) for x in str(args.quantile_levels).split(",") if x.strip()]
    if not quantile_levels:
        raise SystemExit("--quantile_levels 不能为空。")
    for ql in quantile_levels:
        if not (0.0 <= ql <= 1.0):
            raise SystemExit(f"非法分位点 {ql}，必须在 [0,1]。")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_jsonl_paths(input_dir)
    if not paths:
        raise SystemExit(f"未在目录中找到 jsonl: {input_dir}")

    raw_by_step: dict[int, list[dict[str, float]]] = {}
    q_probe_by_step: dict[int, list[dict[str, float]]] = {}
    n_lines = 0
    n_used = 0
    n_bad = 0
    n_dropped_m_eq_1 = 0

    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                if not isinstance(rec, dict):
                    n_bad += 1
                    continue

                step = _safe_int(rec.get("step"))
                if step is None:
                    step = _extract_step_from_filename(p)
                if step is None:
                    n_bad += 1
                    continue

                suffix_rate = _safe_float(rec.get(args.suffix_rate_key))
                m_value = _safe_float(rec.get(args.m_key))
                advantage = _safe_float(rec.get(args.advantage_key))
                q_value = _safe_float(rec.get(args.q_key))
                h_t_value = _safe_float(rec.get(args.h_t_key))
                if advantage is None:
                    n_bad += 1
                    continue
                if args.group_mode == "suffix_rate" and suffix_rate is None:
                    n_bad += 1
                    continue
                if args.group_mode == "m" and m_value is None:
                    n_bad += 1
                    continue

                raw_by_step.setdefault(step, []).append(
                    {
                        "suffix_rate": float(suffix_rate) if suffix_rate is not None else float("nan"),
                        "m_value": float(m_value) if m_value is not None else float("nan"),
                        "advantage": float(advantage),
                    }
                )
                if q_value is not None and h_t_value is not None and suffix_rate is not None:
                    q_probe_by_step.setdefault(step, []).append(
                        {
                            "q": float(q_value),
                            "h_t": float(h_t_value),
                            "suffix_rate": float(suffix_rate),
                        }
                    )
                n_used += 1

    if not raw_by_step:
        raise SystemExit("没有可用记录：请检查字段名或数据内容。")

    steps = sorted(raw_by_step.keys())
    min_group_count = max(1, int(args.min_group_count))

    rows: list[dict[str, float | int]] = []
    for step in steps:
        items = raw_by_step[step]
        sr = np.array([x["suffix_rate"] for x in items], dtype=np.float64)
        m_vals = np.array([x["m_value"] for x in items], dtype=np.float64)
        adv = np.array([x["advantage"] for x in items], dtype=np.float64)

        if args.group_mode == "m":
            valid_mask = np.abs(m_vals - 1.0) > float(args.m_equal_eps)
            n_dropped_m_eq_1 += int(np.sum(~valid_mask))
            sr = sr[valid_mask]
            m_vals = m_vals[valid_mask]
            adv = adv[valid_mask]
            thr = 1.0
            low_mask = m_vals > 1.0
            high_mask = m_vals < 1.0
        elif args.split_mode == "median":
            thr = float(np.median(sr))
            low_mask = sr <= thr
            high_mask = sr > thr
        else:
            thr = float(args.fixed_suffix_threshold)
            low_mask = sr <= thr
            high_mask = sr > thr
        n_low = int(np.sum(low_mask))
        n_high = int(np.sum(high_mask))

        nonzero_mask = np.abs(adv) > float(args.adv_zero_eps)
        low_nonzero_mask = low_mask & nonzero_mask
        high_nonzero_mask = high_mask & nonzero_mask
        n_low_nonzero = int(np.sum(low_nonzero_mask))
        n_high_nonzero = int(np.sum(high_nonzero_mask))
        p_corr_low = (
            float(np.sum(adv[low_nonzero_mask] > 0.0) / n_low_nonzero)
            if n_low_nonzero >= min_group_count
            else float("nan")
        )
        p_corr_high = (
            float(np.sum(adv[high_nonzero_mask] > 0.0) / n_high_nonzero)
            if n_high_nonzero >= min_group_count
            else float("nan")
        )
        e_adv_low = float(np.mean(adv[low_mask])) if n_low >= min_group_count else float("nan")
        e_adv_high = float(np.mean(adv[high_mask])) if n_high >= min_group_count else float("nan")
        low_pos_mask = low_mask & (adv > 0.0)
        high_pos_mask = high_mask & (adv > 0.0)
        low_neg_mask = low_mask & (adv < 0.0)
        high_neg_mask = high_mask & (adv < 0.0)
        n_low_pos = int(np.sum(low_pos_mask))
        n_high_pos = int(np.sum(high_pos_mask))
        n_low_neg = int(np.sum(low_neg_mask))
        n_high_neg = int(np.sum(high_neg_mask))
        e_adv_low_pos = float(np.mean(adv[low_pos_mask])) if n_low_pos >= min_group_count else float("nan")
        e_adv_high_pos = float(np.mean(adv[high_pos_mask])) if n_high_pos >= min_group_count else float("nan")
        e_adv_low_neg = float(np.mean(adv[low_neg_mask])) if n_low_neg >= min_group_count else float("nan")
        e_adv_high_neg = float(np.mean(adv[high_neg_mask])) if n_high_neg >= min_group_count else float("nan")

        n_drop_step = int(np.sum(np.abs(np.array([x["m_value"] for x in items], dtype=np.float64) - 1.0) <= float(args.m_equal_eps)))

        rows.append(
            {
                "step": int(step),
                "split_threshold": float(thr),
                "n_total": int(adv.size),
                "n_dropped_m_eq_1": n_drop_step,
                "n_low_tail": n_low,
                "n_high_head": n_high,
                "n_low_tail_adv_nonzero": n_low_nonzero,
                "n_high_head_adv_nonzero": n_high_nonzero,
                "n_low_tail_adv_pos": n_low_pos,
                "n_high_head_adv_pos": n_high_pos,
                "n_low_tail_adv_neg": n_low_neg,
                "n_high_head_adv_neg": n_high_neg,
                "p_correct_low_tail": p_corr_low,
                "p_correct_high_head": p_corr_high,
                "e_adv_low_tail": e_adv_low,
                "e_adv_high_head": e_adv_high,
                "e_adv_low_tail_adv_pos": e_adv_low_pos,
                "e_adv_high_head_adv_pos": e_adv_high_pos,
                "e_adv_low_tail_adv_neg": e_adv_low_neg,
                "e_adv_high_head_adv_neg": e_adv_high_neg,
            }
        )

    csv_path = output_dir / "fepo_lowtail_highhead_by_step.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "step,split_threshold,n_total,n_dropped_m_eq_1,n_low_tail,n_high_head,"
            "n_low_tail_adv_nonzero,n_high_head_adv_nonzero,"
            "n_low_tail_adv_pos,n_high_head_adv_pos,n_low_tail_adv_neg,n_high_head_adv_neg,"
            "p_correct_low_tail,p_correct_high_head,e_adv_low_tail,e_adv_high_head,"
            "e_adv_low_tail_adv_pos,e_adv_high_head_adv_pos,e_adv_low_tail_adv_neg,e_adv_high_head_adv_neg\n"
        )
        for r in rows:
            f.write(
                f"{r['step']},{r['split_threshold']:.8f},{r['n_total']},{r['n_dropped_m_eq_1']},{r['n_low_tail']},{r['n_high_head']},"
                f"{r['n_low_tail_adv_nonzero']},{r['n_high_head_adv_nonzero']},"
                f"{r['n_low_tail_adv_pos']},{r['n_high_head_adv_pos']},{r['n_low_tail_adv_neg']},{r['n_high_head_adv_neg']},"
                f"{r['p_correct_low_tail']:.8f},{r['p_correct_high_head']:.8f},"
                f"{r['e_adv_low_tail']:.8f},{r['e_adv_high_head']:.8f},"
                f"{r['e_adv_low_tail_adv_pos']:.8f},{r['e_adv_high_head_adv_pos']:.8f},"
                f"{r['e_adv_low_tail_adv_neg']:.8f},{r['e_adv_high_head_adv_neg']:.8f}\n"
            )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_jsonl_files": len(paths),
        "num_lines_total": n_lines,
        "num_lines_used": n_used,
        "num_lines_skipped": n_bad,
        "num_lines_dropped_m_eq_1": int(n_dropped_m_eq_1),
        "split_mode": args.split_mode,
        "group_mode": args.group_mode,
        "fixed_suffix_threshold": float(args.fixed_suffix_threshold),
        "suffix_rate_key": args.suffix_rate_key,
        "m_key": args.m_key,
        "m_equal_eps": float(args.m_equal_eps),
        "advantage_key": args.advantage_key,
        "q_key": args.q_key,
        "h_t_key": args.h_t_key,
        "q_target_high": float(args.q_target_high),
        "q_target_low": float(args.q_target_low),
        "adv_zero_eps": float(args.adv_zero_eps),
        "p_correct_definition": "count(adv>0) / count(|adv|>adv_zero_eps)",
        "min_group_count": int(min_group_count),
        "steps_covered": steps,
    }
    summary_path = output_dir / "fepo_lowtail_highhead_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    q_targets_rows: list[dict[str, float | int]] = []
    for step in steps:
        candidates = q_probe_by_step.get(step, [])
        if not candidates:
            continue
        q_arr = np.array([x["q"] for x in candidates], dtype=np.float64)
        h_arr = np.array([x["h_t"] for x in candidates], dtype=np.float64)
        s_arr = np.array([x["suffix_rate"] for x in candidates], dtype=np.float64)
        idx_high = int(np.argmin(np.abs(q_arr - float(args.q_target_high))))
        idx_low = int(np.argmin(np.abs(q_arr - float(args.q_target_low))))
        q_targets_rows.append(
            {
                "step": int(step),
                "n_candidates_with_q_h_suffix": int(q_arr.size),
                "q_target_high": float(args.q_target_high),
                "q_nearest_high": float(q_arr[idx_high]),
                "abs_diff_high": float(abs(q_arr[idx_high] - float(args.q_target_high))),
                "suffix_rate_at_q_high": float(s_arr[idx_high]),
                "h_t_at_q_high": float(h_arr[idx_high]),
                "q_target_low": float(args.q_target_low),
                "q_nearest_low": float(q_arr[idx_low]),
                "abs_diff_low": float(abs(q_arr[idx_low] - float(args.q_target_low))),
                "suffix_rate_at_q_low": float(s_arr[idx_low]),
                "h_t_at_q_low": float(h_arr[idx_low]),
            }
        )

    q_targets_csv_path = output_dir / "q_target_nearest_suffix_ht_by_step.csv"
    with open(q_targets_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "step,n_candidates_with_q_h_suffix,"
            "q_target_high,q_nearest_high,abs_diff_high,suffix_rate_at_q_high,h_t_at_q_high,"
            "q_target_low,q_nearest_low,abs_diff_low,suffix_rate_at_q_low,h_t_at_q_low\n"
        )
        for r in q_targets_rows:
            f.write(
                f"{r['step']},{r['n_candidates_with_q_h_suffix']},"
                f"{r['q_target_high']:.8f},{r['q_nearest_high']:.8f},{r['abs_diff_high']:.8f},{r['suffix_rate_at_q_high']:.8f},{r['h_t_at_q_high']:.8f},"
                f"{r['q_target_low']:.8f},{r['q_nearest_low']:.8f},{r['abs_diff_low']:.8f},{r['suffix_rate_at_q_low']:.8f},{r['h_t_at_q_low']:.8f}\n"
            )

    quantile_rows: list[dict[str, float | int]] = []
    for step in steps:
        candidates = q_probe_by_step.get(step, [])
        if not candidates:
            continue
        suffix_arr = np.array([x["suffix_rate"] for x in candidates], dtype=np.float64)
        ht_arr = np.array([x["h_t"] for x in candidates], dtype=np.float64)
        row: dict[str, float | int] = {
            "step": int(step),
            "n_candidates_with_q_h_suffix": int(suffix_arr.size),
        }
        for ql in quantile_levels:
            qname = f"{ql:.1f}"
            row[f"suffix_q{qname}"] = float(np.quantile(suffix_arr, ql))
            row[f"h_t_q{qname}"] = float(np.quantile(ht_arr, ql))
        quantile_rows.append(row)

    quantile_csv_path = output_dir / "suffix_ht_quantiles_by_step.csv"
    with open(quantile_csv_path, "w", encoding="utf-8") as f:
        suffix_cols = [f"suffix_q{ql:.1f}" for ql in quantile_levels]
        ht_cols = [f"h_t_q{ql:.1f}" for ql in quantile_levels]
        f.write("step,n_candidates_with_q_h_suffix," + ",".join(suffix_cols + ht_cols) + "\n")
        for r in quantile_rows:
            values = [f"{r['step']}", f"{r['n_candidates_with_q_h_suffix']}"]
            values += [f"{float(r[c]):.8f}" for c in suffix_cols]
            values += [f"{float(r[c]):.8f}" for c in ht_cols]
            f.write(",".join(values) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = np.array([r["step"] for r in rows], dtype=np.float64)
        p_low = np.array([r["p_correct_low_tail"] for r in rows], dtype=np.float64)
        p_high = np.array([r["p_correct_high_head"] for r in rows], dtype=np.float64)
        a_low = np.array([r["e_adv_low_tail"] for r in rows], dtype=np.float64)
        a_high = np.array([r["e_adv_high_head"] for r in rows], dtype=np.float64)
        a_low_pos = np.array([r["e_adv_low_tail_adv_pos"] for r in rows], dtype=np.float64)
        a_high_pos = np.array([r["e_adv_high_head_adv_pos"] for r in rows], dtype=np.float64)
        a_low_neg = np.array([r["e_adv_low_tail_adv_neg"] for r in rows], dtype=np.float64)
        a_high_neg = np.array([r["e_adv_high_head_adv_neg"] for r in rows], dtype=np.float64)

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(xs, p_low, label="P(correct | low-tail)", color="tab:blue", linewidth=2)
        ax1.plot(xs, p_high, label="P(correct | high-head)", color="tab:orange", linewidth=2)
        ax1.set_xlabel("step")
        ax1.set_ylabel("probability")
        ax1.set_title("P(correct | low-tail) vs P(correct | high-head)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(output_dir / "p_correct_lowtail_vs_highhead.png", dpi=160)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(xs, a_low, label="E[advantage | low-tail]", color="tab:green", linewidth=2)
        ax2.plot(xs, a_high, label="E[advantage | high-head]", color="tab:red", linewidth=2)
        ax2.set_xlabel("step")
        ax2.set_ylabel("advantage")
        ax2.set_title("E[advantage | low-tail] vs E[advantage | high-head]")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(output_dir / "e_adv_lowtail_vs_highhead.png", dpi=160)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(xs, a_low_pos, label="E[adv | low-tail, A>0]", color="tab:blue", linewidth=2)
        ax3.plot(xs, a_high_pos, label="E[adv | high-head, A>0]", color="tab:orange", linewidth=2)
        ax3.plot(xs, a_low_neg, label="E[adv | low-tail, A<0]", color="tab:green", linewidth=2)
        ax3.plot(xs, a_high_neg, label="E[adv | high-head, A<0]", color="tab:red", linewidth=2)
        ax3.set_xlabel("step")
        ax3.set_ylabel("advantage")
        ax3.set_title("Conditional E[adv] by low-tail/high-head and sign(A)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(output_dir / "e_adv_four_conditions.png", dpi=160)
        plt.close(fig3)

        if q_targets_rows:
            xq = np.array([r["step"] for r in q_targets_rows], dtype=np.float64)
            suffix_high = np.array([r["suffix_rate_at_q_high"] for r in q_targets_rows], dtype=np.float64)
            suffix_low = np.array([r["suffix_rate_at_q_low"] for r in q_targets_rows], dtype=np.float64)
            ht_high = np.array([r["h_t_at_q_high"] for r in q_targets_rows], dtype=np.float64)
            ht_low = np.array([r["h_t_at_q_low"] for r in q_targets_rows], dtype=np.float64)

            fig4, (ax4, ax5) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax4.plot(xq, suffix_high, label=f"suffix rate @ q~{float(args.q_target_high):.2f}", color="tab:blue", linewidth=2)
            ax4.plot(xq, suffix_low, label=f"suffix rate @ q~{float(args.q_target_low):.2f}", color="tab:orange", linewidth=2)
            ax4.set_ylabel("suffix rate")
            ax4.set_title("Nearest-q suffix entropy metric by step")
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            ax5.plot(xq, ht_high, label=f"h_t @ q~{float(args.q_target_high):.2f}", color="tab:green", linewidth=2)
            ax5.plot(xq, ht_low, label=f"h_t @ q~{float(args.q_target_low):.2f}", color="tab:red", linewidth=2)
            ax5.set_xlabel("step")
            ax5.set_ylabel("h_t")
            ax5.set_title("Nearest-q h_t by step")
            ax5.grid(True, alpha=0.3)
            ax5.legend()

            fig4.tight_layout()
            fig4.savefig(output_dir / "q_target_nearest_suffix_ht_by_step.png", dpi=160)
            plt.close(fig4)

        if quantile_rows:
            x_quant = np.array([r["step"] for r in quantile_rows], dtype=np.float64)

            fig5, ax7 = plt.subplots(figsize=(10, 5))
            for ql in quantile_levels:
                col = f"suffix_q{ql:.1f}"
                y = np.array([r[col] for r in quantile_rows], dtype=np.float64)
                ax7.plot(x_quant, y, linewidth=2, label=f"q={ql:.1f}")
            ax7.set_xlabel("step")
            ax7.set_ylabel("suffix entropy metric")
            ax7.set_title("Suffix metric quantiles by step")
            ax7.grid(True, alpha=0.3)
            ax7.legend(ncol=3, fontsize=9)
            fig5.tight_layout()
            fig5.savefig(output_dir / "suffix_quantiles_by_step.png", dpi=160)
            plt.close(fig5)

            fig6, ax8 = plt.subplots(figsize=(10, 5))
            for ql in quantile_levels:
                col = f"h_t_q{ql:.1f}"
                y = np.array([r[col] for r in quantile_rows], dtype=np.float64)
                ax8.plot(x_quant, y, linewidth=2, label=f"q={ql:.1f}")
            ax8.set_xlabel("step")
            ax8.set_ylabel("h_t")
            ax8.set_title("h_t quantiles by step")
            ax8.grid(True, alpha=0.3)
            ax8.legend(ncol=3, fontsize=9)
            fig6.tight_layout()
            fig6.savefig(output_dir / "h_t_quantiles_by_step.png", dpi=160)
            plt.close(fig6)
    except Exception as e:  # pragma: no cover
        print(f"[warn] 画图失败（可能未安装 matplotlib）: {e}")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote CSV: {q_targets_csv_path}")
    print(f"Wrote CSV: {quantile_csv_path}")
    print(f"Wrote Summary: {summary_path}")
    print(f"Done. steps={len(steps)}, used={n_used}, skipped={n_bad}")


if __name__ == "__main__":
    main()
