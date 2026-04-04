#!/usr/bin/env python3
"""生成 VERL 可用的数学验证集 parquet（MATH-500 + AIME 2024）。

输出字段与 verl/utils/dataset/README.md 一致，便于配合 default_compute_score：
  - MATH-500: data_source 必须为 HuggingFaceH4/MATH-500 -> math_reward.compute_score
  - AIME 2024: data_source 以 aime 开头 -> math_dapo.compute_score

默认从 HuggingFace 下载；需联网。若已在本地缓存，datasets 会直接用缓存。

示例:
  python prepare_verl_math_test_parquets.py --output_dir ~/data/verl_eval
  python prepare_verl_math_test_parquets.py --output_dir ~/data/verl_eval --datasets math500
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from datasets import Dataset, load_dataset

# 与 verl.utils.reward_score.default_compute_score 中的分支一致
MATH500_DATA_SOURCE = "HuggingFaceH4/MATH-500"
AIME_DATA_SOURCE = "aime2024"

DEFAULT_MATH500_REPO = "HuggingFaceH4/MATH-500"
DEFAULT_AIME_REPO = "HuggingFaceH4/aime_2024"

DEFAULT_INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}."
)


def _build_math500_rows(ds, instruction_suffix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(len(ds)):
        ex = ds[idx]
        problem = ex["problem"]
        answer = ex["answer"]
        content = f"{problem.strip()}\n\n{instruction_suffix.strip()}"
        rows.append(
            {
                "data_source": MATH500_DATA_SOURCE,
                "ability": "math",
                "prompt": [{"role": "user", "content": content}],
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {
                    "index": idx,
                    "split": "test",
                    "unique_id": ex.get("unique_id", str(idx)),
                    "subject": ex.get("subject"),
                },
            }
        )
    return rows


def _build_aime_rows(ds, instruction_suffix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # 不同版本 HF 数据集列名可能为 problem/question、answer/solution
    for idx in range(len(ds)):
        ex = ds[idx]
        problem = ex.get("problem") or ex.get("question")
        if problem is None:
            raise KeyError(f"AIME row {idx} has no 'problem' or 'question' field; keys={list(ex.keys())}")
        answer = ex.get("answer")
        if answer is None:
            raise KeyError(f"AIME row {idx} has no 'answer' field; keys={list(ex.keys())}")
        content = f"{str(problem).strip()}\n\n{instruction_suffix.strip()}"
        rows.append(
            {
                "data_source": AIME_DATA_SOURCE,
                "ability": "math",
                "prompt": [{"role": "user", "content": content}],
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {
                    "index": idx,
                    "split": "aime2024",
                    "source_id": ex.get("id", ex.get("unique_id", str(idx))),
                },
            }
        )
    return rows


def _load_math500(split: str = "test"):
    return load_dataset(DEFAULT_MATH500_REPO, split=split)


def _load_aime():
    ds_dict = load_dataset(DEFAULT_AIME_REPO)
    # 常见为单一 train split
    if hasattr(ds_dict, "keys"):
        keys = list(ds_dict.keys())
        for k in ("test", "validation", "train"):
            if k in keys:
                return ds_dict[k]
        return ds_dict[keys[0]]
    return ds_dict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/data/verl_eval"),
        help="输出目录（会写入 math500_test.parquet 与 aime2024_test.parquet）",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="可选: all | math500 | aime24",
    )
    p.add_argument(
        "--instruction_suffix",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="追加在题干后的答题格式说明（需与训练时 prompt 风格一致时可改掉）",
    )
    p.add_argument(
        "--math500_split",
        type=str,
        default="test",
        help="MATH-500 的 split 名（HF 上通常为 test，500 条）",
    )
    args = p.parse_args()

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    mode = args.datasets.lower().strip()
    if mode not in ("all", "math500", "aime24"):
        raise SystemExit("--datasets 只能是 all, math500, aime24")

    if mode in ("all", "math500"):
        print(f"Loading {DEFAULT_MATH500_REPO} split={args.math500_split} ...")
        ds_m = _load_math500(args.math500_split)
        rows_m = _build_math500_rows(ds_m, args.instruction_suffix)
        path_m = os.path.join(out_dir, "math500_test.parquet")
        Dataset.from_list(rows_m).to_parquet(path_m)
        print(f"Wrote MATH-500 ({len(rows_m)} rows) -> {path_m}")

    if mode in ("all", "aime24"):
        print(f"Loading {DEFAULT_AIME_REPO} ...")
        ds_a = _load_aime()
        rows_a = _build_aime_rows(ds_a, args.instruction_suffix)
        path_a = os.path.join(out_dir, "aime2024_test.parquet")
        Dataset.from_list(rows_a).to_parquet(path_a)
        print(f"Wrote AIME 2024 ({len(rows_a)} rows) -> {path_a}")

    print("\n在训练脚本里可设置（按实际生成的文件二选一或两项都写）:")
    parts = []
    if mode in ("all", "math500"):
        parts.append(os.path.join(out_dir, "math500_test.parquet"))
    if mode in ("all", "aime24"):
        parts.append(os.path.join(out_dir, "aime2024_test.parquet"))
    inner = ",".join(repr(x) for x in parts)
    print(f'  data.val_files="[{inner}]"')


if __name__ == "__main__":
    main()
