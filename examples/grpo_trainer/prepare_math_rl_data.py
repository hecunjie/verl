#!/usr/bin/env python3
"""准备 VERL 数学 GRPO 用 parquet。

**默认训练集**：[`open-r1/DAPO-Math-17k-Processed`](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)
（约 1.7 万条，与 DAPO 论文规模一致）。每条保留其 ``prompt`` 字段，将 ``label`` 写入 ``reward_model.ground_truth``。

**测试集**：MATH-500、AIME 2024，题干用与上述数据集 **相同** 的 user 文案模板包裹（含 ``\\boxed`` 与末尾 ``Remember to put...``）。

若需使用字节版百万级数据，加 ``--train_source byted``。

用法（训练集与测试集均从 HuggingFace 在线拉取，需联网）::

  python prepare_math_rl_data.py --output_dir ~/data/math_rl
  python prepare_math_rl_data.py --output_dir ~/data/math_rl --max_train_samples 1024

生成（均在 output_dir 下）：
  - dapo_math_17k_processed_train.parquet  # 默认：Processed 训练（或 byted 时为 dapo_math_byted_train.parquet）
  - math500_test.parquet
  - aime2024_test.parquet

依赖：datasets。"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

# 与 open-r1/DAPO-Math-17k-Processed 中 user 文案一致（用于 MATH-500 / AIME 测试集）
# 注意：须写成 ``{{$Answer}}``，否则 str.format 会把 ``\boxed{$Answer}`` 里的花括号当成占位符。
DAPO_PROCESSED_USER_TEMPLATE = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: \\boxed{{$Answer}} "
    "where $Answer is the answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

# 旧脚本里用于 Byted 全量数据的模板（无末尾 Remember 行）
LEGACY_PROMPT_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: \\boxed{$Answer} "
    "where $Answer is the answer to the problem."
)


def _user_content_legacy(problem: str) -> str:
    return f"{LEGACY_PROMPT_PREFIX.strip()}\n\n{str(problem).strip()}"


def _user_content_dapo_processed(problem: str) -> str:
    return DAPO_PROCESSED_USER_TEMPLATE.format(problem=str(problem).strip())


def _row(
    user_content: str,
    ground_truth: str,
    data_source: str,
    extra_info: dict[str, Any] | None = None,
    ability: str = "math",
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "prompt": [{"role": "user", "content": user_content}],
        "data_source": data_source,
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
    }
    if extra_info is not None:
        r["extra_info"] = extra_info
    return r


def _row_prompt_list(
    prompt: list[dict[str, Any]],
    ground_truth: str,
    data_source: str,
    extra_info: dict[str, Any] | None = None,
    ability: str = "math",
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "prompt": prompt,
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


def _ensure_prompt_list(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [{"role": "user", "content": prompt.strip()}]
    if isinstance(prompt, list):
        return prompt
    raise ValueError(f"unsupported prompt type: {type(prompt)}")


def _ground_truth_from_processed(ex: dict[str, Any]) -> str:
    if ex.get("label") is not None:
        return _gt(ex["label"])
    rm = ex.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        return _gt(rm["ground_truth"])
    raise ValueError(f"no label/reward_model.ground_truth in example keys={list(ex.keys())}")


def _load_dapo_processed_hf() -> Dataset:
    """HF 上该数据集子集名为 ``all``，split 为 ``train``（约 17.4k 行）。"""
    last_err: Exception | None = None
    for factory in (
        lambda: load_dataset("open-r1/DAPO-Math-17k-Processed", "all", split="train"),
        lambda: load_dataset("open-r1/DAPO-Math-17k-Processed", split="train"),
    ):
        try:
            raw = factory()
            if isinstance(raw, DatasetDict):
                return _pick_split(raw)
            return raw
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise RuntimeError(
        "无法从 HuggingFace 加载 open-r1/DAPO-Math-17k-Processed（请检查网络、代理与 HF_TOKEN 限流）。"
    ) from last_err


def iter_dapo_processed_train(max_train_samples: int | None = None):
    ds = _load_dapo_processed_hf()
    n = len(ds) if max_train_samples is None else min(int(max_train_samples), len(ds))
    for idx in range(n):
        ex = ds[idx]
        prompt = _ensure_prompt_list(ex["prompt"])
        gt = _ground_truth_from_processed(ex)
        extra = ex.get("extra_info")
        if not isinstance(extra, dict):
            extra = {}
        extra = {**extra, "index": idx}
        yield _row_prompt_list(
            prompt,
            gt,
            "open-r1/DAPO-Math-17k-Processed",
            extra_info=extra,
            ability=str(ex.get("ability", "math")),
        )


def _first_user_prompt(prompt: list) -> str:
    for m in prompt:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m["content"]).strip()
    raise ValueError("prompt 中无 user 消息")


def iter_byted_dapo_train(max_train_samples: int | None = None):
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", "default", split="train")
    n = len(ds) if max_train_samples is None else min(int(max_train_samples), len(ds))
    for idx in range(n):
        ex = ds[idx]
        prob = _first_user_prompt(ex["prompt"])
        yield _row(
            _user_content_legacy(prob),
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
            _user_content_dapo_processed(str(ex["problem"]).strip()),
            _gt(ex["answer"]),
            "HuggingFaceH4/MATH-500",
            extra_info={
                "index": idx,
                "split": "test",
                "unique_id": ex.get("unique_id", str(idx)),
                "subject": ex.get("subject"),
            },
        )


def _pick_split(ds_dict: DatasetDict):
    keys = list(ds_dict.keys())
    for k in ("test", "validation", "train"):
        if k in keys:
            return ds_dict[k]
    return ds_dict[keys[0]]


def iter_aime24_test():
    raw = load_dataset("HuggingFaceH4/aime_2024")
    ds = _pick_split(raw) if isinstance(raw, DatasetDict) else raw
    for idx in range(len(ds)):
        ex = ds[idx]
        p = ex.get("problem") or ex.get("question")
        if p is None:
            raise KeyError(f"AIME idx={idx} keys={list(ex.keys())}")
        a = ex.get("answer")
        if a is None:
            raise KeyError(f"AIME idx={idx} no answer")
        yield _row(
            _user_content_dapo_processed(str(p).strip()),
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
    ap.add_argument(
        "--train_source",
        type=str,
        choices=("processed", "byted"),
        default="processed",
        help="processed=open-r1/DAPO-Math-17k-Processed；byted=BytedTsinghua-SIA 百万级全量",
    )
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="仅导出训练集前 N 条",
    )
    args = ap.parse_args()
    out = os.path.expanduser(args.output_dir)
    os.makedirs(out, exist_ok=True)

    if args.train_source == "processed":
        f_train = os.path.join(out, "dapo_math_17k_processed_train.parquet")

        def gen_train():
            yield from iter_dapo_processed_train(args.max_train_samples)

        print("1/3 open-r1/DAPO-Math-17k-Processed (train) ...")
        n1 = _write_parquet(f_train, gen_train)
    else:
        f_train = os.path.join(out, "dapo_math_byted_train.parquet")

        def gen_byted():
            yield from iter_byted_dapo_train(args.max_train_samples)

        print("1/3 BytedTsinghua-SIA/DAPO-Math-17k (train, 大规模) ...")
        n1 = _write_parquet(f_train, gen_byted)

    print(f"    -> {n1} rows, {f_train}")

    f_m500 = os.path.join(out, "math500_test.parquet")
    f_aime = os.path.join(out, "aime2024_test.parquet")

    print("2/3 MATH-500 (test, DAPO-Processed 同款 prompt) ...")
    n2 = _write_parquet(f_m500, iter_math500_test)
    print(f"    -> {n2} rows, {f_m500}")

    print("3/3 AIME 2024 (test, 同上) ...")
    n3 = _write_parquet(f_aime, iter_aime24_test)
    print(f"    -> {n3} rows, {f_aime}")

    print("\n训练 / 验证可设为:")
    print(f'  data.train_files="{f_train}"')
    print(f'  data.val_files="[\'{f_m500}\',\'{f_aime}\']"')


if __name__ == "__main__":
    main()
