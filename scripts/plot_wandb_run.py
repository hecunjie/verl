#!/usr/bin/env python3
# Copyright 2025 the script authors
"""
从 W&B 云端 API 拉取单次 run 的标量历史，并导出高质量 matplotlib 图。

用法示例
--------
  export WANDB_API_KEY=...   # 或 wandb login 已配置

  # Run 页面 URL 形如 https://wandb.ai/<entity>/<project>/runs/<run_id>
  python plot_wandb_run.py --entity myteam --project verl --run-id abc123xyz

  # 或一行路径
  python plot_wandb_run.py --run-path myteam/verl/abc123xyz

  # 只画指定指标（逗号分隔，支持子串匹配多条）
  python plot_wandb_run.py --run-path myteam/verl/abc123 \\
      --metrics "val/reward,actor/loss"

  # 叠加在同一张图（适合量级接近的曲线）
  python plot_wandb_run.py --run-path myteam/verl/abc123 \\
      --layout overlay --metrics "train/loss,critic/loss"

  # 多 run 对比：每个指标单独一张图；图例用 --methods（与 --runs 一一对应）
  python plot_wandb_run.py --entity myteam --project verl \\
      --runs abc111 def222 ghi333 jkl444 \\
      --methods FEPO GRPO DAPO GTPO \\
      --metrics "m1,m2,m3,m4" --out-dir ./plots_compare \\
      --max-step 500

  # 横轴从约 30 step 起画，且刻度只标 100 的倍数（不出现 30）：
  #   --min-step 30 --x-tick-multiple 100

# 具名方法 FEPO / GTPO / GRPO 使用脚本内固定配色（红 / 蓝 / 绿），其它方法仍走默认调色盘。

依赖: pip install wandb matplotlib pandas
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator

# Okabe–Ito 色盲友好调色（高对比；未命中具名方法时使用）
_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# 多 run 对比：具名方法固定色（不区分大小写；键为方法名去掉首尾空格后的小写）
_METHOD_COLORS: dict[str, str] = {
    "fepo": "#D32F2F",  # red
    "gtpo": "#1976D2",  # blue
    "grpo": "#2E7D32",  # green
}


def _color_for_method(method_label: str, fallback_index: int) -> str:
    key = method_label.strip().lower()
    if key in _METHOD_COLORS:
        return _METHOD_COLORS[key]
    return _PALETTE[fallback_index % len(_PALETTE)]


def _setup_matplotlib_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": False,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.35,
            "lines.markersize": 3,
        }
    )


def short_metric_ylabel(metric: str) -> str:
    """
    将完整 wandb key 压成短纵轴标签，例如 mean@4、major@4。
    """
    m = metric.strip().lower()
    # Special-case for AIME25 visualization: show best@4 curve as mean@4 on y-axis label.
    if "aime2025" in m and ("best@4" in m or "best_at_4" in m or "best_at4" in m):
        return "mean@4"

    parts = metric.rstrip("/").split("/")
    if len(parts) >= 2 and parts[-1] == "mean" and "@" in parts[-2]:
        core = parts[-2]
    else:
        core = parts[-1]
    if core.startswith("maj@"):
        return "major" + core[3:]
    return core


def _is_aime25_best4_metric(metric: str) -> bool:
    m = metric.strip().lower()
    return "aime2025" in m and ("best@4" in m or "best_at_4" in m or "best_at4" in m)


def _find_aime25_best4_metric(available_metrics: Iterable[str]) -> str | None:
    """Find one AIME2025 best@4 metric key from available columns/keys."""
    for m in available_metrics:
        if _is_aime25_best4_metric(m):
            return m
    return None


def _append_extra_aime25_best4(metric_list: list[str], available_metrics: Iterable[str]) -> list[str]:
    """If not explicitly requested, append one AIME2025 best@4 metric when present."""
    out = list(metric_list)
    if any(_is_aime25_best4_metric(m) for m in out):
        return out
    extra = _find_aime25_best4_metric(available_metrics)
    if extra and extra not in out:
        out.append(extra)
    return out


def _aime25_best4_key_candidates() -> list[str]:
    """Common W&B key variants for AIME2025 best@4."""
    return [
        "val-core/aime2025/acc/best@4",
        "val-core/aime2025/acc/best_at_4",
        "val-core/aime2025/acc/best_at4",
        "val-core/aime2025/acc/best@4/mean",
        "val-core/aime2025/acc/best_at_4/mean",
        "val-core/aime2025/acc/best_at4/mean",
    ]


def _apply_step_range_mask(
    df: pd.DataFrame,
    x_col: str,
    min_step: int | None,
    max_step: int | None,
) -> pd.DataFrame:
    """按横轴列裁剪：可选最小 step（含）与最大 step（含）。"""
    xs = pd.to_numeric(df[x_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if min_step is not None and min_step > 0:
        mask &= xs >= float(min_step)
    if max_step is not None and max_step > 0:
        mask &= xs <= float(max_step)
    return df.loc[mask].copy()


def _set_xticks_multiple_of(
    ax,
    x_min: float,
    x_max: float,
    base: float,
) -> None:
    """
    横轴主刻度仅为 base 的倍数（落在数据范围内）；不强制出现小于 base 的刻度（例如起点 30 时不会出现 30）。
    base<=0 时不修改刻度（交给 matplotlib 默认）。
    """
    if base <= 0 or not (math.isfinite(x_min) and math.isfinite(x_max)):
        return
    lo, hi = min(x_min, x_max), max(x_min, x_max)
    first = math.ceil(lo / base) * base
    last = math.floor(hi / base) * base
    if first <= last:
        ticks = np.arange(first, last + base * 0.5, base)
        ax.set_xticks(ticks)
    ax.xaxis.set_minor_locator(NullLocator())


def _strip_internal_columns(df: pd.DataFrame) -> list[str]:
    skip = {"_step", "_runtime", "_timestamp", "system/", "wandb/"}
    cols: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if any(c.startswith(s) for s in ("system/", "wandb/")):
            continue
        if df[c].dtype.kind not in "iufb":
            continue
        cols.append(c)
    return cols


def _match_metrics(all_metrics: list[str], patterns: list[str] | None) -> list[str]:
    if not patterns:
        return all_metrics
    selected: list[str] = []
    for p in patterns:
        p = p.strip()
        if not p:
            continue
        regex = False
        if p.startswith("re:"):
            regex = True
            p = p[3:]
        if regex:
            r = re.compile(p)
            for m in all_metrics:
                if r.search(m) and m not in selected:
                    selected.append(m)
        else:
            for m in all_metrics:
                if p in m or m == p:
                    if m not in selected:
                        selected.append(m)
    return selected


def fetch_history(
    entity: str,
    project: str,
    run_id: str,
    keys: Iterable[str] | None,
    samples: int | None,
    x_col: str,
) -> pd.DataFrame:
    import wandb

    timeout = int(os.environ.get("WANDB_API_TIMEOUT", "300"))
    api = wandb.Api(timeout=timeout)
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    key_list = list(keys) if keys else None
    if key_list is not None:
        extra = {x_col, "_step", "_runtime", "_timestamp"}
        key_list = sorted(set(key_list) | {k for k in extra if k})

    if samples is not None and samples > 0:
        # 注意：history 默认曾长期为 500 行；显式 samples 用于「最近 N 条」等场景
        df = run.history(samples=samples, keys=key_list, pandas=True)
    else:
        # 全量：scan_history（大数据 run 会慢、占内存）
        rows = list(run.scan_history(keys=key_list))
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df


def plot_subplots(
    df: pd.DataFrame,
    metrics: list[str],
    x_col: str,
    outfile: str,
    x_label: str = "step",
    min_step: int | None = None,
    max_step: int | None = None,
    x_tick_multiple: float = 0.0,
) -> None:
    n = len(metrics)
    fig_h = min(2.8 * n + 0.6, 26)
    fig, axes = plt.subplots(n, 1, figsize=(10.2, fig_h), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    dfp = _apply_step_range_mask(df, x_col, min_step, max_step)
    xs = pd.to_numeric(dfp[x_col], errors="coerce").to_numpy()
    for ax, m, color in zip(axes, metrics, _PALETTE * (1 + n // len(_PALETTE))):
        ys = pd.to_numeric(dfp[m], errors="coerce")
        ax.plot(xs, ys, color=color, label=m, solid_capstyle="round", linewidth=1.35)
        ax.set_ylabel(short_metric_ylabel(m), color=color, fontweight="medium")
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel(x_label)
    if x_tick_multiple > 0 and xs.size:
        x_lo, x_hi = float(np.nanmin(xs)), float(np.nanmax(xs))
        _set_xticks_multiple_of(axes[-1], x_lo, x_hi, x_tick_multiple)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def _safe_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s.replace("@", "_at_"))[:180]


def plot_multi_run_one_metric(
    run_frames: list[tuple[str, pd.DataFrame]],
    metric: str,
    x_col: str,
    outfile: str,
    *,
    min_step: int | None,
    max_step: int | None,
    x_label: str = "step",
    line_width: float = 1.15,
    x_tick_multiple: float = 0.0,
) -> None:
    """同一指标、多条曲线；run_frames 为 (method 名, df)，图例为方法名；每条曲线同色系画最大值水平参考线。"""
    fig, ax = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.85, alpha=0.9)
    any_line = False
    y_label = short_metric_ylabel(metric)
    x_lo, x_hi = float("inf"), float("-inf")
    for i, (method_label, df) in enumerate(run_frames):
        if metric not in df.columns:
            print(f"warn: metric {metric!r} missing for method {method_label!r}, skip", file=sys.stderr)
            continue
        dfp = _apply_step_range_mask(df, x_col, min_step, max_step)
        xs = pd.to_numeric(dfp[x_col], errors="coerce").to_numpy()
        ys = pd.to_numeric(dfp[metric], errors="coerce").to_numpy()
        if xs.size:
            x_lo = min(x_lo, float(np.nanmin(xs)))
            x_hi = max(x_hi, float(np.nanmax(xs)))
        color = _color_for_method(method_label, i)
        ax.plot(xs, ys, color=color, label=method_label, solid_capstyle="round", linewidth=line_width)
        finite = ys[np.isfinite(ys)]
        if finite.size > 0:
            ymax = float(np.nanmax(finite))
            ax.axhline(
                ymax,
                color=color,
                linestyle="--",
                linewidth=line_width * 0.85,
                alpha=0.92,
                zorder=3,
            )
        any_line = True
    if not any_line:
        plt.close(fig)
        raise ValueError(f"no data plotted for metric {metric!r}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, fontweight="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", framealpha=0.95, edgecolor="#CCCCCC")
    if x_tick_multiple > 0 and math.isfinite(x_lo) and math.isfinite(x_hi):
        _set_xticks_multiple_of(ax, x_lo, x_hi, x_tick_multiple)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    df: pd.DataFrame,
    metrics: list[str],
    x_col: str,
    outfile: str,
    x_label: str = "step",
    min_step: int | None = None,
    max_step: int | None = None,
    x_tick_multiple: float = 0.0,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.85, alpha=0.9)
    dfp = _apply_step_range_mask(df, x_col, min_step, max_step)
    xs = pd.to_numeric(dfp[x_col], errors="coerce").to_numpy()
    for i, m in enumerate(metrics):
        color = _PALETTE[i % len(_PALETTE)]
        ys = pd.to_numeric(dfp[m], errors="coerce")
        ax.plot(xs, ys, color=color, label=short_metric_ylabel(m), solid_capstyle="round", linewidth=1.35)
    ax.set_xlabel(x_label)
    ax.set_ylabel("value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", framealpha=0.92, edgecolor="#CCCCCC")
    if x_tick_multiple > 0 and xs.size:
        _set_xticks_multiple_of(ax, float(np.nanmin(xs)), float(np.nanmax(xs)), x_tick_multiple)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull W&B run history via API and plot.")
    parser.add_argument("--run-path", type=str, default="", help="entity/project/run_id (单 run)")
    parser.add_argument("--entity", type=str, default="", help="W&B entity (team or user)")
    parser.add_argument("--project", type=str, default="", help="W&B project")
    parser.add_argument("--run-id", type=str, default="", help="W&B run id (单 run)")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        metavar="RUN_ID",
        help="多个 run id：与 --entity --project 联用，每个 --metrics 中的指标各输出一张对比图",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="与 --runs 一一对应的方法名（仅多 run 模式必填，用作图例，不写 run id）",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Comma-separated metric names or substrings; prefix with re: for regex",
    )
    parser.add_argument(
        "--layout",
        choices=("multi", "overlay"),
        default="multi",
        help="multi: one subplot per metric; overlay: single axis (scales must be comparable)",
    )
    parser.add_argument(
        "--x-axis",
        type=str,
        default="_step",
        help="数据列名作为横轴（默认 _step；与 --x-label 显示文字无关）",
    )
    parser.add_argument(
        "--x-label",
        type=str,
        default="step",
        help="横轴显示标签（默认 step）",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=0,
        help="仅绘制横轴 step 不大于此值的点（截断）；0 表示不限制",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=0,
        help="仅绘制横轴 step 不小于该值的点（左侧截断，如 30）；0 表示从数据起点",
    )
    parser.add_argument(
        "--x-tick-multiple",
        type=float,
        default=0.0,
        help="横轴刻度仅为该数的倍数（如 100，则只显示 100,200,...，不会出现 30）；0 表示用 matplotlib 默认刻度",
    )
    parser.add_argument("--samples", type=int, default=0, help="Max rows from API (0 = full scan_history)")
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Optional rolling-mean window (steps) applied before plotting; 0 = off",
    )
    parser.add_argument("--output", type=str, default="", help="Output PNG path")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="多 run 对比时输出目录（默认当前目录下 wandb_compare_<project>）",
    )
    parser.add_argument("--csv", type=str, default="", help="Optional: save fetched table to CSV")
    args = parser.parse_args()

    _setup_matplotlib_style()
    x_col = args.x_axis
    x_label = args.x_label
    max_step = args.max_step if args.max_step > 0 else None
    min_step = args.min_step if args.min_step > 0 else None
    x_tick_multiple = float(args.x_tick_multiple) if args.x_tick_multiple and args.x_tick_multiple > 0 else 0.0
    samples = args.samples if args.samples and args.samples > 0 else None

    # ----- 多 run 对比 -----
    if args.runs is not None:
        if not args.entity or not args.project:
            print("error: --runs 需要同时指定 --entity 与 --project", file=sys.stderr)
            return 2
        if args.run_path or args.run_id:
            print("error: 多 run 模式下不要同时使用 --run-path 或 --run-id", file=sys.stderr)
            return 2
        if not args.methods:
            print("error: 多 run 模式必须提供 --methods，且与 --runs 数量一致", file=sys.stderr)
            return 2
        if len(args.methods) != len(args.runs):
            print(
                f"error: --methods 数量 ({len(args.methods)}) 与 --runs ({len(args.runs)}) 不一致",
                file=sys.stderr,
            )
            return 2
        metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
        if not metric_list:
            print("error: --runs 模式下必须提供 --metrics（逗号分隔，每个指标输出一张图）", file=sys.stderr)
            return 2
        entity, project = args.entity, args.project
        # Also request AIME2025 best@4 candidates so auto-extra plotting can detect them.
        keys = list({*metric_list, *_aime25_best4_key_candidates(), x_col, "_step", "_runtime", "_timestamp"})
        run_frames: list[tuple[str, pd.DataFrame]] = []
        for rid, method in zip(args.runs, args.methods):
            df = fetch_history(entity, project, rid, keys=keys, samples=samples, x_col=x_col)
            if df.empty:
                print(f"warn: empty history for run {rid} ({method}), skip", file=sys.stderr)
                continue
            if args.smooth and args.smooth > 1:
                df = df.copy()
                for m in metric_list:
                    if m in df.columns:
                        df[m] = (
                            pd.to_numeric(df[m], errors="coerce")
                            .rolling(window=args.smooth, min_periods=1)
                            .mean()
                        )
            run_frames.append((method, df))

        if not run_frames:
            print("error: no valid runs loaded", file=sys.stderr)
            return 1

        if x_col not in run_frames[0][1].columns:
            print(f"error: x-axis column {x_col!r} not in dataframe", file=sys.stderr)
            return 1

        # Auto-add AIME2025 best@4 plot if the metric exists in loaded runs.
        available_multi_cols: set[str] = set()
        for _method, _df in run_frames:
            available_multi_cols.update(str(c) for c in _df.columns)
        metric_list = _append_extra_aime25_best4(metric_list, available_multi_cols)

        out_dir = args.out_dir or f"wandb_compare_{project}"
        os.makedirs(out_dir, exist_ok=True)

        for metric in metric_list:
            out_png = os.path.join(out_dir, f"compare_{_safe_filename(metric)}.png")
            try:
                plot_multi_run_one_metric(
                    run_frames,
                    metric,
                    x_col,
                    out_png,
                    min_step=min_step,
                    max_step=max_step,
                    x_label=x_label,
                    x_tick_multiple=x_tick_multiple,
                )
                print(f"saved figure: {out_png}")
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
        return 0

    # ----- 单 run -----
    if args.run_path:
        parts = args.run_path.strip("/").split("/")
        if len(parts) != 3:
            print("error: --run-path must be entity/project/run_id", file=sys.stderr)
            return 2
        entity, project, run_id = parts
    else:
        if not args.entity or not args.project or not args.run_id:
            print("error: 单 run 请使用 --run-path，或同时提供 --entity --project --run-id", file=sys.stderr)
            return 2
        entity, project, run_id = args.entity, args.project, args.run_id

    patterns = [p for p in args.metrics.split(",") if p.strip()] if args.metrics else None
    # 始终拉全量标量键再在本地筛选，避免把子串误判成完整列名导致缺列
    df = fetch_history(entity, project, run_id, keys=None, samples=samples, x_col=x_col)

    if df.empty:
        print("error: empty history (check run id / permissions / keys)", file=sys.stderr)
        return 1

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"saved table: {args.csv}")

    if x_col not in df.columns:
        print(f"error: x-axis column {x_col!r} not in dataframe columns", file=sys.stderr)
        return 1

    all_numeric = _strip_internal_columns(df)
    metrics = _match_metrics(all_numeric, patterns)
    # Auto-add AIME2025 best@4 figure when available.
    metrics = _append_extra_aime25_best4(metrics, all_numeric)
    if not metrics:
        print("error: no metrics matched; available numeric columns sample:", file=sys.stderr)
        print(", ".join(all_numeric[:40]), file=sys.stderr)
        return 1

    if args.smooth and args.smooth > 1:
        for m in metrics:
            df[m] = pd.to_numeric(df[m], errors="coerce").rolling(window=args.smooth, min_periods=1).mean()

    out = args.output or f"wandb_plot_{project}_{run_id}.png"
    if args.layout == "multi":
        plot_subplots(
            df,
            metrics,
            x_col,
            out,
            x_label=x_label,
            min_step=min_step,
            max_step=max_step,
            x_tick_multiple=x_tick_multiple,
        )
    else:
        plot_overlay(
            df,
            metrics,
            x_col,
            out,
            x_label=x_label,
            min_step=min_step,
            max_step=max_step,
            x_tick_multiple=x_tick_multiple,
        )
    print(f"saved figure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
