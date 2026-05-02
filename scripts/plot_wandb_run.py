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

  # 多 run 对比：每个指标单独一张图，每条曲线一个 run（需同一 entity/project）
  python plot_wandb_run.py --entity myteam --project verl \\
      --runs abc111 def222 ghi333 jkl444 \\
      --metrics "m1,m2,m3,m4" --out-dir ./plots_compare

依赖: pip install wandb matplotlib pandas
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Okabe–Ito 色盲友好调色（高对比）
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


def _setup_matplotlib_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "lines.markersize": 4,
        }
    )


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


def plot_subplots(df: pd.DataFrame, metrics: list[str], x_col: str, outfile: str, title: str) -> None:
    n = len(metrics)
    fig_h = min(3.2 * n, 36)
    fig, axes = plt.subplots(n, 1, figsize=(11, fig_h), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    xs = df[x_col].to_numpy()
    for ax, m, color in zip(axes, metrics, _PALETTE * (1 + n // len(_PALETTE))):
        ys = pd.to_numeric(df[m], errors="coerce")
        ax.plot(xs, ys, color=color, label=m, solid_capstyle="round")
        ax.set_ylabel(m, color=color, fontweight="medium")
        ax.tick_params(axis="y", labelcolor=color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel(x_col)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def _safe_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s.replace("@", "_at_"))[:180]


def plot_multi_run_one_metric(
    run_frames: list[tuple[str, pd.DataFrame]],
    metric: str,
    x_col: str,
    outfile: str,
    title: str,
) -> None:
    """同一指标、多条 run 曲线（不同颜色）。"""
    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    any_line = False
    for i, (label, df) in enumerate(run_frames):
        if metric not in df.columns:
            print(f"warn: metric {metric!r} missing in run {label}, skip", file=sys.stderr)
            continue
        color = _PALETTE[i % len(_PALETTE)]
        xs = pd.to_numeric(df[x_col], errors="coerce")
        ys = pd.to_numeric(df[metric], errors="coerce")
        ax.plot(xs, ys, color=color, label=label, solid_capstyle="round")
        any_line = True
    if not any_line:
        plt.close(fig)
        raise ValueError(f"no data plotted for metric {metric!r}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric, fontweight="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", framealpha=0.95, edgecolor="#CCCCCC", title="run id")
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(df: pd.DataFrame, metrics: list[str], x_col: str, outfile: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    xs = df[x_col].to_numpy()
    for i, m in enumerate(metrics):
        color = _PALETTE[i % len(_PALETTE)]
        ys = pd.to_numeric(df[m], errors="coerce")
        ax.plot(xs, ys, color=color, label=m, solid_capstyle="round")
    ax.set_xlabel(x_col)
    ax.set_ylabel("value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", framealpha=0.92, edgecolor="#CCCCCC")
    ax.set_title(title, fontsize=14, fontweight="bold")
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
        help="Column for x-axis (default _step; try _runtime for wall time)",
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
    samples = args.samples if args.samples and args.samples > 0 else None

    # ----- 多 run 对比 -----
    if args.runs is not None:
        if not args.entity or not args.project:
            print("error: --runs 需要同时指定 --entity 与 --project", file=sys.stderr)
            return 2
        if args.run_path or args.run_id:
            print("error: 多 run 模式下不要同时使用 --run-path 或 --run-id", file=sys.stderr)
            return 2
        metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
        if not metric_list:
            print("error: --runs 模式下必须提供 --metrics（逗号分隔，每个指标输出一张图）", file=sys.stderr)
            return 2
        entity, project = args.entity, args.project
        keys = list({*metric_list, x_col, "_step", "_runtime", "_timestamp"})
        run_frames: list[tuple[str, pd.DataFrame]] = []
        for rid in args.runs:
            df = fetch_history(entity, project, rid, keys=keys, samples=samples, x_col=x_col)
            if df.empty:
                print(f"warn: empty history for run {rid}, skip", file=sys.stderr)
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
            run_frames.append((rid, df))

        if not run_frames:
            print("error: no valid runs loaded", file=sys.stderr)
            return 1

        if x_col not in run_frames[0][1].columns:
            print(f"error: x-axis column {x_col!r} not in dataframe", file=sys.stderr)
            return 1

        out_dir = args.out_dir or f"wandb_compare_{project}"
        os.makedirs(out_dir, exist_ok=True)

        for metric in metric_list:
            title = f"{entity}/{project}\n{metric}"
            if args.smooth and args.smooth > 1:
                title += f" (rolling mean, window={args.smooth})"
            out_png = os.path.join(out_dir, f"compare_{_safe_filename(metric)}.png")
            try:
                plot_multi_run_one_metric(run_frames, metric, x_col, out_png, title)
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
    if not metrics:
        print("error: no metrics matched; available numeric columns sample:", file=sys.stderr)
        print(", ".join(all_numeric[:40]), file=sys.stderr)
        return 1

    if args.smooth and args.smooth > 1:
        for m in metrics:
            df[m] = pd.to_numeric(df[m], errors="coerce").rolling(window=args.smooth, min_periods=1).mean()

    title = f"{entity}/{project} — {run_id}"
    if args.smooth and args.smooth > 1:
        title += f" (rolling mean, window={args.smooth})"
    out = args.output or f"wandb_plot_{project}_{run_id}.png"
    if args.layout == "multi":
        plot_subplots(df, metrics, x_col, out, title)
    else:
        plot_overlay(df, metrics, x_col, out, title)
    print(f"saved figure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
