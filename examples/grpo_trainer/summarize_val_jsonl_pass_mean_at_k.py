#!/usr/bin/env python3
"""从 ``trainer.validation_data_dir`` 下的 ``<step>.jsonl`` 汇总 mean@k / pass@k（严格定义）。

VERL 控制台里的 ``val-core/.../acc/mean@N`` 与 ``best@N/mean`` 来自 ``metric_utils.process_validation_metrics``
（后者为 bootstrap 意义下的 max，**不等价**于论文里常用的 unbiased pass@k）。

本脚本按 **prompt 文本 ``input``** 分组（与 val dump 一致；同题 32 条样本 ``input`` 相同），对每组：

- **mean@k**：取该组前 ``k`` 条样本的 ``acc``（或 ``score``）的算术平均；再对所有组取 macro 平均。
- **pass@k（严格）**：若前 ``k`` 条中任一条正确（acc/score > 0.5）则该组记 1，否则 0；再对组平均。
- **pass@k（无偏估计量）**：若该组总样本数为 ``n``、正确数为 ``c``，使用
  ``1 - C(n-c,k)/C(n,k)``（与 ``examples/entropy_ce/recompute_passk_metrics_from_merged.py`` 相同）。

用法::

  python summarize_val_jsonl_pass_mean_at_k.py \\
    --jsonl /path/to/0.jsonl \\
    --k 32

依赖：Python 3.10+，标准库。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    if n <= 0:
        return float("nan")
    kk = min(max(int(k), 1), int(n))
    cc = min(max(int(c), 0), int(n))
    if n - cc < kk:
        return 1.0
    prod = 1.0
    for i in range(kk):
        prod *= float(n - cc - i) / float(n - i)
    return 1.0 - prod


def _as_correct(x: Any) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    return 1.0 if float(x) > 0.5 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=str, required=True, help="验证 dump 的 jsonl，例如 validation_data_dir/0.jsonl")
    ap.add_argument("--k", type=int, default=32, help="前 k 条样本用于 mean@k / pass@k（严格）")
    ap.add_argument(
        "--key",
        type=str,
        default="acc",
        choices=("acc", "score"),
        help="从每行读取正确性所用的字段（math_verify 一般为 acc 或 score）",
    )
    args = ap.parse_args()
    path = Path(args.jsonl).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")

    k = max(int(args.k), 1)
    key = str(args.key)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            inp = rec.get("input")
            if inp is None:
                raise SystemExit("record missing 'input' field")
            groups[str(inp)].append(rec)

    strict_pass: list[float] = []
    unbiased_pass: list[float] = []
    mean_at_k_list: list[float] = []

    for _prompt, rows in groups.items():
        if key not in rows[0]:
            if key == "acc" and "score" in rows[0]:
                key = "score"
            else:
                raise SystemExit(f"records missing '{args.key}' (first keys: {list(rows[0].keys())})")

        corr = [_as_correct(r[key]) for r in rows]
        n = len(corr)
        c = int(sum(1 for x in corr if x > 0.5))
        kk = min(k, n)
        head = corr[:kk]
        mean_at_k_list.append(float(np.mean(head)) if head else float("nan"))
        strict_pass.append(1.0 if any(x > 0.5 for x in head) else 0.0)
        unbiased_pass.append(_pass_at_k_unbiased(n=n, c=c, k=k))

    print(
        json.dumps(
            {
                "jsonl": str(path),
                "num_groups": len(groups),
                "k": k,
                "key_used": key,
                "mean_at_k_macro": float(np.nanmean(mean_at_k_list)) if mean_at_k_list else float("nan"),
                "pass_at_k_strict_macro": float(np.mean(strict_pass)) if strict_pass else float("nan"),
                "pass_at_k_unbiased_macro": float(np.nanmean(unbiased_pass)) if unbiased_pass else float("nan"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
