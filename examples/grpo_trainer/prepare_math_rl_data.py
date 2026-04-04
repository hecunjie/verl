#!/usr/bin/env python3
"""从 HuggingFace 下载 DAPO-Math-17k（训练）、MATH-500（测）、AIME 2024（测），并写成 VERL 可直接用的 parquet。

每条样本：``DEFAULT_PROMPT_PREFIX`` + 换行 + 题干，与统一数学 RL 格式一致。

仅需一个参数::

  python prepare_math_rl_data.py --output_dir ~/data/math_rl

生成（均在 output_dir 下）：
  - dapo_math_17k_train.parquet   # 训练
  - math500_test.parquet         # 验证
  - aime2024_test.parquet        # 验证

需联网；依赖 datasets。"""
from __future__ import annotations

import argparse
import os
from typing import Any

from datasets import Dataset, load_dataset

DEFAULT_PROMPT_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: \\boxed{$Answer} "
    "where $Answer is the answer to the problem."
)


def _user_content(problem: str) -> str:
    return f"{DEFAULT_PROMPT_PREFIX.strip()}\n\n{str(problem).strip()}"


def _row(
    problem: str,
    ground_truth: str,
    data_source: str,
    extra_info: dict[str, Any] | None = None,
    ability: str = "math",
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "prompt": [{"role": "user", "content": _user_content(problem)}],
        "data_source": data_source,
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
    }
    if extra_info is not None:
        r["extra_info"] = extra_info
    return r


def _gt(gt: Any) -> str:
    if gt is None:
        raise ValueError("missing ground_truth")
    if isinstance(gt, list):
        return str(gt[0]) if len(gt) == 1 else str(gt)
    return str(gt)


def _first_user_prompt(prompt: list) -> str:
    for m in prompt:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m["content"]).strip()
    raise ValueError("prompt 中无 user 消息")


def iter_dapo_train():
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", "default", split="train")
    for idx in range(len(ds)):
        ex = ds[idx]
        prob = _first_user_prompt(ex["prompt"])
        yield _row(
            prob,
            _gt(ex["reward_model"]["ground_truth"]),
            ex.get("data_source") or "math_dapo",
            extra_info={**(ex.get("extra_info") or {}), "index": idx},
            ability=str(ex.get("ability", "math")),
        )


def iter_math500_test():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for idx in range(len(ds)):
        ex = ds[idx]
        yield _row(
            str(ex["problem"]).strip(),
            _gt(ex["answer"]),
            "HuggingFaceH4/MATH-500",
            extra_info={
                "index": idx,
                "split": "test",
                "unique_id": ex.get("unique_id", str(idx)),
                "subject": ex.get("subject"),
            },
        )


def _pick_split(ds_dict):
    keys = list(ds_dict.keys())
    for k in ("test", "validation", "train"):
        if k in keys:
            return ds_dict[k]
    return ds_dict[keys[0]]


def iter_aime24_test():
    raw = load_dataset("HuggingFaceH4/aime_2024")
    ds = raw if hasattr(raw, "__len__") else _pick_split(raw)
    for idx in range(len(ds)):
        ex = ds[idx]
        p = ex.get("problem") or ex.get("question")
        if p is None:
            raise KeyError(f"AIME idx={idx} keys={list(ex.keys())}")
        a = ex.get("answer")
        if a is None:
            raise KeyError(f"AIME idx={idx} no answer")
        yield _row(
            str(p).strip(),
            _gt(a),
            "aime2024",
            extra_info={"index": idx, "split": "aime2024", "source_id": ex.get("id", str(idx))},
        )


def _write_parquet(path: str, gen):
    ds = Dataset.from_generator(gen)
    ds.to_parquet(path)
    return len(ds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", type=str, required=True, help="输出目录（将自动创建）")
    args = ap.parse_args()
    out = os.path.expanduser(args.output_dir)
    os.makedirs(out, exist_ok=True)

    f_train = os.path.join(out, "dapo_math_17k_train.parquet")
    f_m500 = os.path.join(out, "math500_test.parquet")
    f_aime = os.path.join(out, "aime2024_test.parquet")

    print("1/3 DAPO-Math-17k (train) ...")
    n1 = _write_parquet(f_train, iter_dapo_train)
    print(f"    -> {n1} rows, {f_train}")

    print("2/3 MATH-500 (test) ...")
    n2 = _write_parquet(f_m500, iter_math500_test)
    print(f"    -> {n2} rows, {f_m500}")

    print("3/3 AIME 2024 ...")
    n3 = _write_parquet(f_aime, iter_aime24_test)
    print(f"    -> {n3} rows, {f_aime}")

    print("\n训练 / 验证可设为:")
    print(f'  data.train_files="{f_train}"')
    print(f'  data.val_files="[\'{f_m500}\',\'{f_aime}\']"')


if __name__ == "__main__":
    main()
