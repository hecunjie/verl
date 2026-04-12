#!/usr/bin/env python3
"""按 entropy_t 阈值过滤 pair_bias_merged.jsonl，比较 correct / wrong 上 bias_t 与 delta_hat_t 的分布。

用法示例::

    python analyze_pair_bias_entropy_filter.py \\
        --input /path/to/pair_bias_merged.jsonl \\
        --entropy_threshold 1.0 \\
        --output_dir ./pair_bias_filtered_plots

也可同时给多个阈值::

    python analyze_pair_bias_entropy_filter.py -i pair_bias_merged.jsonl \\
        --entropy_threshold 0.5 1.0 1.5 --output_dir ./out

若希望用「熵分位数」而非绝对阈值（只保留熵最高的前 q 比例记录）::

    python analyze_pair_bias_entropy_filter.py -i pair_bias_merged.jsonl \\
        --entropy_quantile 0.25 --output_dir ./out
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _split_by_group(rows: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[float]]:
    h_c: list[float] = []
    h_w: list[float] = []
    b_c: list[float] = []
    b_w: list[float] = []
    d_c: list[float] = []
    d_w: list[float] = []
    for r in rows:
        g = r.get("group")
        if g not in ("correct", "wrong"):
            continue
        try:
            h = float(r["entropy_t"])
            b = float(r["bias_t"])
            d = float(r["delta_hat_t"])
        except (KeyError, TypeError, ValueError):
            continue
        if g == "correct":
            h_c.append(h)
            b_c.append(b)
            d_c.append(d)
        else:
            h_w.append(h)
            b_w.append(b)
            d_w.append(d)
    return b_c, b_w, d_c, d_w


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "iqr": float("nan"),
            "range": float("nan"),
            "mad": float("nan"),
        }
    xs_sorted = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)
    std = math.sqrt(var) if n > 1 else 0.0

    def _pct(p: float) -> float:
        if n == 1:
            return xs_sorted[0]
        k = (n - 1) * p
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return xs_sorted[lo]
        return xs_sorted[lo] * (hi - k) + xs_sorted[hi] * (k - lo)

    p25, p50, p75 = _pct(0.25), _pct(0.5), _pct(0.75)
    iqr = p75 - p25
    med = p50
    mad = sum(abs(x - med) for x in xs) / n
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "min": min(xs),
        "max": max(xs),
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "iqr": iqr,
        "range": max(xs) - min(xs),
        "mad": mad,
    }


def _filter_by_entropy_threshold(
    rows: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            h = float(r["entropy_t"])
        except (KeyError, TypeError, ValueError):
            continue
        if h >= threshold:
            out.append(r)
    return out


def _filter_by_entropy_quantile(rows: list[dict[str, Any]], q: float) -> list[dict[str, Any]]:
    """只保留 entropy_t 全局最高的前 q 比例（q=0.25 表示熵最高的 25% 记录；并列时可能略多）。"""
    pairs: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        try:
            h = float(r["entropy_t"])
        except (KeyError, TypeError, ValueError):
            continue
        pairs.append((h, r))
    if not pairs:
        return []
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    k = max(1, int(math.ceil(q * n)))
    thr = pairs[n - k][0]
    return [r for h, r in pairs if h >= thr]


def _plot_hist2(
    correct: list[float],
    wrong: list[float],
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5))
    all_v = correct + wrong
    if not all_v:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        lo, hi = min(all_v), max(all_v)
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        edges = np.linspace(lo, hi, bins + 1)

        c_hist, _ = np.histogram(correct, bins=edges, density=True)
        w_hist, _ = np.histogram(wrong, bins=edges, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        width = (edges[1] - edges[0]) * 0.36
        ax.bar(centers - width / 2, c_hist, width=width, label=f"correct (n={len(correct)})", alpha=0.75)
        ax.bar(centers + width / 2, w_hist, width=width, label=f"wrong (n={len(wrong)})", alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_violin(
    correct: list[float],
    wrong: list[float],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [correct, wrong]
    labels = [f"correct\nn={len(correct)}", f"wrong\nn={len(wrong)}"]
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter pair_bias jsonl by entropy_t and plot distributions.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/mnt/tidal-alsh01/dataset/zeus/hecunjie/entropy_check/ds_distill_1.5B_bias_correct_wrong_first_sentence_qwen3-4b-8gpu/pair_bias_merged.jsonl",
        help="pair_bias_merged.jsonl 路径",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="输出目录：保存图与 summary json",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        nargs="*",
        default=[],
        help="只保留 entropy_t >= 该阈值的记录；可多个，每个阈值一套图与统计。未与 --entropy_quantile 同时指定时，默认 1.0",
    )
    parser.add_argument(
        "--entropy_quantile",
        type=float,
        default=None,
        help="若设置，则忽略 entropy_threshold：只保留 entropy_t 最高的前 q 比例（如 0.25=最高 25%%）",
    )
    parser.add_argument("--bins", type=int, default=40, help="直方图分箱数")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.is_file():
        raise SystemExit(f"input not found: {in_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _load_records(in_path)
    if not all_rows:
        raise SystemExit("empty jsonl")

    modes: list[tuple[str, list[dict[str, Any]]]] = []
    if args.entropy_quantile is not None:
        q = float(args.entropy_quantile)
        if not (0 < q <= 1):
            raise SystemExit("--entropy_quantile must be in (0, 1]")
        filtered = _filter_by_entropy_quantile(all_rows, q)
        modes.append((f"quantile_top_{q:g}", filtered))
    else:
        thresholds = list(args.entropy_threshold) if args.entropy_threshold else [1.0]
        for thr in thresholds:
            filtered = _filter_by_entropy_threshold(all_rows, float(thr))
            modes.append((f"entropy_ge_{thr:g}", filtered))

    import json as json_mod

    full_summary: dict[str, Any] = {"input": str(in_path), "n_total_lines": len(all_rows)}

    for tag, filtered in modes:
        b_c, b_w, d_c, d_w = _split_by_group(filtered)
        sub: dict[str, Any] = {
            "n_filtered": len(filtered),
            "bias_t": {
                "correct": _stats(b_c),
                "wrong": _stats(b_w),
            },
            "delta_hat_t": {
                "correct": _stats(d_c),
                "wrong": _stats(d_w),
            },
        }
        for key, bc, bw in (("bias_t", b_c, b_w), ("delta_hat_t", d_c, d_w)):
            sc, sw = _stats(bc), _stats(bw)
            cmp: dict[str, Any] = {}
            if sc["std"] and sc["std"] > 0:
                cmp["std_ratio_wrong_over_correct"] = sw["std"] / sc["std"]
            else:
                cmp["std_ratio_wrong_over_correct"] = None
            if sc["iqr"] and sc["iqr"] > 0:
                cmp["iqr_ratio_wrong_over_correct"] = sw["iqr"] / sc["iqr"]
            else:
                cmp["iqr_ratio_wrong_over_correct"] = None
            if sc["range"] and sc["range"] > 0:
                cmp["range_ratio_wrong_over_correct"] = sw["range"] / sc["range"]
            else:
                cmp["range_ratio_wrong_over_correct"] = None
            sub[key]["spread_compare"] = cmp

        full_summary[tag] = sub

        tag_dir = out_dir / tag.replace(".", "_")
        tag_dir.mkdir(parents=True, exist_ok=True)

        _plot_hist2(b_c, b_w, f"{tag}: bias_t", "bias_t", tag_dir / "hist_bias_t.png", args.bins)
        _plot_hist2(d_c, d_w, f"{tag}: delta_hat_t", "delta_hat_t", tag_dir / "hist_delta_hat_t.png", args.bins)
        _plot_violin(b_c, b_w, f"{tag}: bias_t", "bias_t", tag_dir / "violin_bias_t.png")
        _plot_violin(d_c, d_w, f"{tag}: delta_hat_t", "delta_hat_t", tag_dir / "violin_delta_hat_t.png")

        with open(tag_dir / "summary.json", "w", encoding="utf-8") as f:
            json_mod.dump(sub, f, ensure_ascii=False, indent=2)

        print(f"\n=== {tag} ===")
        print(f"n_filtered={len(filtered)}")
        for label, bc, bw in (("bias_t", b_c, b_w), ("delta_hat_t", d_c, d_w)):
            sc, sw = _stats(bc), _stats(bw)
            print(f"  [{label}] correct: n={int(sc['n'])}, std={sc['std']:.6f}, IQR={sc['iqr']:.6f}")
            print(f"  [{label}] wrong:   n={int(sw['n'])}, std={sw['std']:.6f}, IQR={sw['iqr']:.6f}")
            if sc["std"]:
                print(f"  [{label}] std ratio (wrong/correct) = {sw['std']/sc['std']:.4f}")
            if sc["iqr"]:
                print(f"  [{label}] IQR ratio (wrong/correct) = {sw['iqr']/sc['iqr']:.4f}")

    out_json = out_dir / "full_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json_mod.dump(full_summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote: {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
