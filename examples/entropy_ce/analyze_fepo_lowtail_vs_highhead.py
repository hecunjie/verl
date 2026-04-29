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
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_jsonl_paths(input_dir)
    if not paths:
        raise SystemExit(f"未在目录中找到 jsonl: {input_dir}")

    raw_by_step: dict[int, list[dict[str, float]]] = {}
    n_lines = 0
    n_used = 0
    n_bad = 0

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
            valid_mask = m_vals != 1.0
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

        p_corr_low = float(np.mean(adv[low_mask] > 0.0)) if n_low >= min_group_count else float("nan")
        p_corr_high = float(np.mean(adv[high_mask] > 0.0)) if n_high >= min_group_count else float("nan")
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

        rows.append(
            {
                "step": int(step),
                "split_threshold": float(thr),
                "n_total": int(adv.size),
                "n_low_tail": n_low,
                "n_high_head": n_high,
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
            "step,split_threshold,n_total,n_low_tail,n_high_head,"
            "n_low_tail_adv_pos,n_high_head_adv_pos,n_low_tail_adv_neg,n_high_head_adv_neg,"
            "p_correct_low_tail,p_correct_high_head,e_adv_low_tail,e_adv_high_head,"
            "e_adv_low_tail_adv_pos,e_adv_high_head_adv_pos,e_adv_low_tail_adv_neg,e_adv_high_head_adv_neg\n"
        )
        for r in rows:
            f.write(
                f"{r['step']},{r['split_threshold']:.8f},{r['n_total']},{r['n_low_tail']},{r['n_high_head']},"
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
        "split_mode": args.split_mode,
        "group_mode": args.group_mode,
        "fixed_suffix_threshold": float(args.fixed_suffix_threshold),
        "suffix_rate_key": args.suffix_rate_key,
        "m_key": args.m_key,
        "advantage_key": args.advantage_key,
        "p_correct_definition": "advantage > 0",
        "min_group_count": int(min_group_count),
        "steps_covered": steps,
    }
    summary_path = output_dir / "fepo_lowtail_highhead_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

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
    except Exception as e:  # pragma: no cover
        print(f"[warn] 画图失败（可能未安装 matplotlib）: {e}")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Summary: {summary_path}")
    print(f"Done. steps={len(steps)}, used={n_used}, skipped={n_bad}")


if __name__ == "__main__":
    main()
