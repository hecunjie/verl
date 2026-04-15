#!/usr/bin/env python3
"""按归一化位置分桶，比较正确/错误回答的「后缀熵和」曲线并绘图。

定义（对每个 response 的 token 位置 i，0-based）：
  H[i] 为该步的 token 熵（输入里已给好的序列）
  S[i] = sum_{j=i}^{n-1} H[j]  （从位置 i 到末尾的熵之和）

将长度 n 的序列在位置轴上均匀切成 ``num_bins`` 个格子：
  位置 i 落入 bin b = min(num_bins-1, floor(i * num_bins / n))

每个样本在每个 bin 内：对该 bin 内所有 i 的 S[i] 取平均，得到长度 num_bins 的向量；
再在样本组内（正确 / 错误）对同一 bin 取均值与标准差，并画图。

输入 JSONL 每行需包含：
  - ``entropies``: list[float]  （与 response 等长）
  - 正确性标签，任选其一：
      ``is_correct`` (bool) / ``acc`` (0/1) / ``label`` (bool 或 0/1)

或每行包含 ``rollouts``: list[dict]，每个 dict 含 ``entropies`` 与上述标签之一。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def _truthy_label(obj: dict[str, Any]) -> bool | None:
    if "is_correct" in obj:
        v = obj["is_correct"]
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(int(v))
    if "acc" in obj:
        v = obj["acc"]
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v) > 0.5
    if "label" in obj:
        v = obj["label"]
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(int(v))
    return None


def _iter_samples_from_record(rec: dict[str, Any]) -> Iterator[tuple[list[float], bool]]:
    label = _truthy_label(rec)
    if label is not None and "entropies" in rec:
        ent = rec["entropies"]
        if isinstance(ent, list) and ent:
            yield [float(x) for x in ent], bool(label)
        return
    rollouts = rec.get("rollouts")
    if isinstance(rollouts, list):
        for r in rollouts:
            if not isinstance(r, dict):
                continue
            lb = _truthy_label(r)
            ent = r.get("entropies")
            if lb is None or not isinstance(ent, list) or not ent:
                continue
            yield [float(x) for x in ent], bool(lb)


def _suffix_entropy_sum(h: np.ndarray) -> np.ndarray:
    """S[i] = sum_{j=i}^{n-1} h[j]."""
    n = int(h.size)
    if n == 0:
        return h
    return np.flip(np.cumsum(np.flip(h, axis=0)), axis=0)


def _per_trajectory_bin_means(h: np.ndarray, num_bins: int) -> np.ndarray | None:
    """Return shape (num_bins,) with nan where bin empty for this trajectory."""
    n = int(h.size)
    if n == 0:
        return None
    s = _suffix_entropy_sum(h)
    acc = np.zeros(num_bins, dtype=np.float64)
    cnt = np.zeros(num_bins, dtype=np.int64)
    for i in range(n):
        b = min(num_bins - 1, int(i * num_bins // n))
        acc[b] += float(s[i])
        cnt[b] += 1
    out = np.full(num_bins, np.nan, dtype=np.float64)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask].astype(np.float64)
    return out


def _nanmean_stack(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack rows (each num_bins), return mean, std, count per bin across samples."""
    if not rows:
        z = np.zeros(0)
        return z, z, z
    m = np.stack(rows, axis=0)
    valid = np.isfinite(m)
    cnt = valid.sum(axis=0).astype(np.float64)
    cnt_safe = np.maximum(cnt, 1.0)
    total = np.nansum(m, axis=0)
    mean = total / cnt_safe
    mean = np.where(cnt > 0, mean, np.nan)
    var = np.nansum((m - mean) ** 2 * valid, axis=0) / np.maximum(cnt - 1.0, 1.0)
    var = np.where(cnt > 1, var, np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std, cnt


def main() -> None:
    p = argparse.ArgumentParser(description="Suffix-entropy curve by normalized bins: correct vs wrong.")
    p.add_argument("--input", action="append", required=True, help="JSONL path (可重复传入多个文件).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_bins", type=int, default=100)
    p.add_argument("--max_lines", type=int, default=0, help="<=0 表示读全文件.")
    args = p.parse_args()

    num_bins = max(2, int(args.num_bins))
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    correct_rows: list[np.ndarray] = []
    wrong_rows: list[np.ndarray] = []
    n_skipped = 0
    n_lines = 0
    max_lines = int(args.max_lines)

    for path_str in args.input:
        if max_lines > 0 and n_lines >= max_lines:
            break
        path = Path(path_str).expanduser().resolve()
        if not path.is_file():
            print(f"[warn] skip missing file: {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if max_lines > 0 and n_lines >= max_lines:
                    break
                n_lines += 1  # 非空 JSON 行计数（用于 --max_lines）
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_skipped += 1
                    continue
                if not isinstance(rec, dict):
                    n_skipped += 1
                    continue
                for ent_list, is_ok in _iter_samples_from_record(rec):
                    h = np.array(ent_list, dtype=np.float64)
                    if not np.all(np.isfinite(h)):
                        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
                    row = _per_trajectory_bin_means(h, num_bins)
                    if row is None or not np.any(np.isfinite(row)):
                        n_skipped += 1
                        continue
                    if is_ok:
                        correct_rows.append(row)
                    else:
                        wrong_rows.append(row)

    mean_c, std_c, cnt_c = _nanmean_stack(correct_rows)
    mean_w, std_w, cnt_w = _nanmean_stack(wrong_rows)

    x = np.arange(num_bins, dtype=np.float64)

    summary = {
        "num_bins": int(num_bins),
        "n_input_lines": int(n_lines),
        "n_skipped_records": int(n_skipped),
        "n_correct_trajectories": int(len(correct_rows)),
        "n_wrong_trajectories": int(len(wrong_rows)),
        "per_bin_count_correct": cnt_c.tolist() if cnt_c.size else [],
        "per_bin_count_wrong": cnt_w.tolist() if cnt_w.size else [],
        "mean_suffix_entropy_correct": mean_c.tolist() if mean_c.size else [],
        "mean_suffix_entropy_wrong": mean_w.tolist() if mean_w.size else [],
        "std_suffix_entropy_correct": std_c.tolist() if std_c.size else [],
        "std_suffix_entropy_wrong": std_w.tolist() if std_w.size else [],
    }
    with open(out_dir / "suffix_entropy_bin_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV for downstream
    csv_path = out_dir / "suffix_entropy_bin_curves.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "bin,x_norm,mean_correct,std_correct,n_correct,mean_wrong,std_wrong,n_wrong\n"
        )
        for b in range(num_bins):
            xn = (b + 0.5) / float(num_bins)
            f.write(
                f"{b},{xn:.6f},"
                f"{mean_c[b] if mean_c.size else math.nan:.8f},"
                f"{std_c[b] if std_c.size else math.nan:.8f},"
                f"{int(cnt_c[b]) if cnt_c.size else 0},"
                f"{mean_w[b] if mean_w.size else math.nan:.8f},"
                f"{std_w[b] if std_w.size else math.nan:.8f},"
                f"{int(cnt_w[b]) if cnt_w.size else 0}\n"
            )

    plot_path = out_dir / "suffix_entropy_bin_curve_correct_vs_wrong.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        xn = (x + 0.5) / float(num_bins)
        if mean_c.size:
            ax.plot(xn, mean_c, label=f"correct (n={len(correct_rows)})", color="tab:blue", linewidth=2)
            if np.any(np.isfinite(std_c)):
                lo = mean_c - std_c / np.sqrt(np.maximum(cnt_c, 1.0))
                hi = mean_c + std_c / np.sqrt(np.maximum(cnt_c, 1.0))
                ax.fill_between(xn, lo, hi, color="tab:blue", alpha=0.2)
        if mean_w.size:
            ax.plot(xn, mean_w, label=f"wrong (n={len(wrong_rows)})", color="tab:orange", linewidth=2)
            if np.any(np.isfinite(std_w)):
                lo = mean_w - std_w / np.sqrt(np.maximum(cnt_w, 1.0))
                hi = mean_w + std_w / np.sqrt(np.maximum(cnt_w, 1.0))
                ax.fill_between(xn, lo, hi, color="tab:orange", alpha=0.2)
        ax.set_xlabel("normalized position (bin center)")
        ax.set_ylabel("mean suffix sum of token entropies (within bin)")
        ax.set_title("Suffix entropy curve: correct vs wrong (uniform position bins)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote plot: {plot_path}", file=sys.stderr)
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib unavailable or plot failed: {e}", file=sys.stderr)

    print(f"Wrote summary: {out_dir / 'suffix_entropy_bin_summary.json'}", file=sys.stderr)
    print(f"Wrote csv: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
