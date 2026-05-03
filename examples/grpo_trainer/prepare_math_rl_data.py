#!/usr/bin/env python3
"""准备 VERL 数学 GRPO 用 parquet。

**默认训练集**：[`open-r1/DAPO-Math-17k-Processed`](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)
（约 1.7 万条，与 DAPO 论文规模一致）。训练样本会统一为 DAPO prompt 指引格式（末行要求 ``Answer: \\boxed{...}``），并将 ``label`` 写入 ``reward_model.ground_truth``。

**测试集**：MATH-500、AIME 2024、AIME 2025、AMC 2023（``math-ai/amc23``）、OlympiadBench（``math-ai/olympiadbench``）、
Minerva-Math（``svc-huggingface/minerva-math``）、GPQA-D（diamond）。
其中数学类题干用与上述数据集 **相同** 的 user 文案模板包裹（含 ``\\boxed`` 与末尾 ``Remember to put...``）；
GPQA-D 会转为四选一格式（末行 ``Answer: X``，X∈{A,B,C,D}）。
另外可选：GSM8K（test）、MATH-lighteval（test），同样统一为 DAPO 风格 prompt。
并支持从 MATH-lighteval 筛出 ``math_hard``（默认 ``level>=5``）单独保存。

若需使用字节版百万级数据，加 ``--train_source byted``。

用法（训练集与测试集均从 HuggingFace 在线拉取，需联网）::

  python prepare_math_rl_data.py --output_dir ~/data/math_rl
  python prepare_math_rl_data.py --output_dir ~/data/math_rl --max_train_samples 1024

生成（均在 output_dir 下）：
  - dapo_math_17k_processed_train.parquet  # 默认：Processed 训练（或 byted 时为 dapo_math_byted_train.parquet）
  - math500_test.parquet
  - aime2024_test.parquet
  - aime2025_test.parquet
  - amc23_test.parquet
  - olympiadbench_test.parquet
  - minerva_math_test.parquet
  - gsm8k_test.parquet
  - math_lighteval_test.parquet
  - math_hard_test.parquet
  - gpqa_d_test.parquet

依赖：datasets。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
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


def _user_content_gpqa_d(question: str, choices: list[str]) -> str:
    opts = []
    letters = ["A", "B", "C", "D"]
    for i, c in enumerate(choices[:4]):
        opts.append(f"{letters[i]}. {str(c).strip()}")
    choice_block = "\n".join(opts)
    return (
        "Answer the following multiple-choice question.\n\n"
        f"{str(question).strip()}\n\n"
        f"{choice_block}\n\n"
        "Think step by step, then put the final option letter on its own line in the format:\n"
        "Answer: X"
    )


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


def _normalize_processed_prompt(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure processed-train prompt uses the DAPO instruction format.

    Some rows in open-r1/DAPO-Math-17k-Processed only contain a raw problem statement.
    This causes model outputs to miss the expected `Answer:` format and hurts rule-based scoring.
    """
    normalized: list[dict[str, Any]] = []
    for m in prompt:
        if not isinstance(m, dict):
            normalized.append(m)
            continue
        if m.get("role") != "user":
            normalized.append(m)
            continue
        content = str(m.get("content", "")).strip()
        has_instruction = (
            "Answer:" in content
            and "\\boxed" in content
            and "Remember to put your answer on its own line after" in content
        )
        if not has_instruction:
            content = _user_content_dapo_processed(content)
        normalized.append({**m, "content": content})
    return normalized


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
        prompt = _normalize_processed_prompt(_ensure_prompt_list(ex["prompt"]))
        gt = _ground_truth_from_processed(ex)
        extra = ex.get("extra_info")
        if not isinstance(extra, dict):
            extra = {}
        extra = {**extra, "index": idx}
        yield _row_prompt_list(
            prompt,
            gt,
            "math_dapo",
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
            "math_dapo",
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
            extra_info={"index": idx, "split": "aime2024", "source_id": str(ex.get("id", str(idx)))},
        )


def iter_aime25_test():
    raw = load_dataset("math-ai/aime25")
    ds = _pick_split(raw) if isinstance(raw, DatasetDict) else raw
    for idx in range(len(ds)):
        ex = ds[idx]
        p = ex.get("problem") or ex.get("question")
        if p is None:
            raise KeyError(f"AIME25 idx={idx} keys={list(ex.keys())}")
        a = ex.get("answer")
        if a is None:
            raise KeyError(f"AIME25 idx={idx} no answer")
        yield _row(
            _user_content_dapo_processed(str(p).strip()),
            _gt(a),
            "aime2025",
            extra_info={"index": idx, "split": "aime2025", "source_id": str(ex.get("id", str(idx)))},
        )


def iter_amc23_test():
    """AMC 2023 竞赛题（40 条），见 `math-ai/amc23`。"""
    raw = load_dataset("math-ai/amc23")
    ds = _pick_split(raw) if isinstance(raw, DatasetDict) else raw
    for idx in range(len(ds)):
        ex = ds[idx]
        p = ex.get("question")
        a = ex.get("answer")
        if p is None or a is None:
            raise KeyError(f"AMC23 idx={idx} keys={list(ex.keys())}")
        yield _row(
            _user_content_dapo_processed(str(p).strip()),
            _gt(a),
            "math-ai/amc23",
            extra_info={
                "index": idx,
                "split": "test",
                "source_id": str(ex.get("id", str(idx))),
                "url": ex.get("url"),
            },
        )


def iter_olympiadbench_test():
    """OlympiadBench 文本题测试集，见 `math-ai/olympiadbench`（test）。"""
    raw = load_dataset("math-ai/olympiadbench")
    ds = _pick_split(raw) if isinstance(raw, DatasetDict) else raw
    for idx in range(len(ds)):
        ex = ds[idx]
        p = ex.get("question")
        a = ex.get("final_answer")
        if p is None or a is None:
            raise KeyError(f"OlympiadBench idx={idx} keys={list(ex.keys())}")
        yield _row(
            _user_content_dapo_processed(str(p).strip()),
            _gt(a),
            "math-ai/olympiadbench",
            extra_info={
                "index": idx,
                "split": "test",
                "source_id": str(ex.get("id", str(idx))),
                "modality": ex.get("modality"),
                "difficulty": ex.get("difficulty"),
                "language": ex.get("language"),
                "subject": ex.get("subject"),
            },
        )


def iter_minerva_math_test(dataset_id: str = "svc-huggingface/minerva-math"):
    """Minerva-Math（大学/研究生难度 STEM 数学题），标准答案从 `solution` 中最后一个 ``\\boxed{}`` 抽取。"""
    raw = load_dataset(dataset_id)
    ds = _pick_split(raw) if isinstance(raw, DatasetDict) else raw
    for idx in range(len(ds)):
        ex = ds[idx]
        p = ex.get("problem")
        sol = ex.get("solution")
        if p is None or sol is None:
            raise KeyError(f"Minerva-Math idx={idx} keys={list(ex.keys())}")
        gt = _extract_math_final_answer(sol)
        if not str(gt).strip():
            raise ValueError(f"Minerva-Math idx={idx}: could not extract ground truth from solution")
        yield _row(
            _user_content_dapo_processed(str(p).strip()),
            gt,
            "svc-huggingface/minerva-math",
            extra_info={
                "index": idx,
                "split": "test",
                "course_type": ex.get("type"),
                "hf_dataset": dataset_id,
                "minerva_idx": ex.get("idx"),
            },
        )


def _extract_gsm8k_final_answer(ans: Any) -> str:
    s = str(ans or "").strip()
    # GSM8K commonly stores rationale + final answer in "#### 42" form.
    if "####" in s:
        return s.split("####")[-1].strip()
    return s


def _extract_last_boxed_content(text: str) -> str | None:
    """Extract content of the last \\boxed{...} with simple brace matching."""
    s = str(text or "")
    key = "\\boxed{"
    pos = s.rfind(key)
    if pos < 0:
        return None
    i = pos + len(key)
    depth = 1
    start = i
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i].strip()
        i += 1
    return None


def _extract_math_final_answer(ans: Any) -> str:
    """Normalize math ground-truth to final answer when solution text is provided."""
    s = str(ans or "").strip()
    if not s:
        return s
    boxed = _extract_last_boxed_content(s)
    if boxed:
        return boxed
    # Fallbacks for common formats
    if "####" in s:
        return s.split("####")[-1].strip()
    if "Answer:" in s:
        return s.split("Answer:")[-1].strip()
    return s


def iter_gsm8k_test():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for idx in range(len(ds)):
        ex = ds[idx]
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            raise KeyError(f"GSM8K idx={idx} keys={list(ex.keys())}")
        yield _row(
            _user_content_dapo_processed(str(q).strip()),
            _extract_gsm8k_final_answer(a),
            "math",
            extra_info={"index": idx, "split": "gsm8k_test"},
        )


def iter_math_lighteval_test():
    raw = load_dataset("DigitalLearningGmbH/MATH-lighteval")
    ds = _pick_split(raw) if isinstance(raw, DatasetDict) else raw
    for idx in range(len(ds)):
        ex = ds[idx]
        p = ex.get("problem") or ex.get("question")
        a = ex.get("answer") or ex.get("solution")
        if p is None or a is None:
            raise KeyError(f"MATH-lighteval idx={idx} keys={list(ex.keys())}")
        yield _row(
            _user_content_dapo_processed(str(p).strip()),
            _extract_math_final_answer(a),
            "DigitalLearningGmbH/MATH-lighteval",
            extra_info={
                "index": idx,
                "split": "test",
                "subject": ex.get("subject"),
                "level": ex.get("level"),
            },
        )


def _level_as_int(x: Any) -> int | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    # common patterns: "5", "Level 5", "level 5"
    for tok in s.replace("_", " ").split():
        if tok.isdigit():
            return int(tok)
    if s.isdigit():
        return int(s)
    return None


def iter_math_hard_test(min_level: int = 5):
    raw = load_dataset("DigitalLearningGmbH/MATH-lighteval")
    ds = _pick_split(raw) if isinstance(raw, DatasetDict) else raw
    out_idx = 0
    for idx in range(len(ds)):
        ex = ds[idx]
        lv = _level_as_int(ex.get("level"))
        if lv is None or int(lv) < int(min_level):
            continue
        p = ex.get("problem") or ex.get("question")
        a = ex.get("answer") or ex.get("solution")
        if p is None or a is None:
            continue
        yield _row(
            _user_content_dapo_processed(str(p).strip()),
            _extract_math_final_answer(a),
            "math_dapo",
            extra_info={
                "index": out_idx,
                "source_index": idx,
                "split": "test",
                "eval_set": "math_hard",
                "subject": ex.get("subject"),
                "level": ex.get("level"),
                "min_level_filter": int(min_level),
            },
        )
        out_idx += 1


def _load_gpqa_d_hf(
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
) -> Dataset:
    """Load GPQA-Diamond from HF with fallback dataset/config names."""
    last_err: Exception | None = None
    attempt_errors: list[str] = []
    factories: list = []
    if dataset_name:
        if dataset_config:
            factories.append(
                lambda: load_dataset(
                    dataset_name,
                    dataset_config,
                    split=dataset_split,
                )
            )
        else:
            factories.append(lambda: load_dataset(dataset_name, split=dataset_split))
    factories.extend(
        (
            lambda: load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=dataset_split),
            lambda: load_dataset("Idavidrein/gpqa", "diamond", split=dataset_split),
            lambda: load_dataset("Idavidrein/gpqa", split=dataset_split),
            lambda: load_dataset("openai/gpqa", "diamond", split=dataset_split),
            lambda: load_dataset("openai/gpqa", split=dataset_split),
        )
    )
    for factory in factories:
        try:
            raw = factory()
            if isinstance(raw, DatasetDict):
                return _pick_split(raw)
            return raw
        except Exception as e:
            last_err = e
            attempt_errors.append(f"{type(e).__name__}: {e}")
            continue
    assert last_err is not None
    raise RuntimeError(
        "无法从 HuggingFace 加载 GPQA-D（diamond）数据集。"
        "该数据集通常是 gated（需在 HF 页面同意条款，并在环境中设置 HF_TOKEN）。"
        f" 最近错误: {type(last_err).__name__}: {last_err}"
        + (f" | 尝试记录: {' || '.join(attempt_errors[:3])}" if attempt_errors else "")
    ) from last_err


def _first_present(ex: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return None


def _extract_gpqa_d_row(ex: dict[str, Any], idx: int) -> tuple[str, list[str], str]:
    question = _first_present(ex, ["question", "Question", "problem", "query", "prompt"])
    if question is None:
        raise KeyError(f"GPQA-D idx={idx} no question field, keys={list(ex.keys())}")

    options: list[str] | None = None
    # schema 1: explicit A/B/C/D columns
    a = _first_present(ex, ["A", "a", "option_a", "choice_a"])
    b = _first_present(ex, ["B", "b", "option_b", "choice_b"])
    c = _first_present(ex, ["C", "c", "option_c", "choice_c"])
    d = _first_present(ex, ["D", "d", "option_d", "choice_d"])
    if all(x is not None for x in (a, b, c, d)):
        options = [str(a), str(b), str(c), str(d)]

    # schema 2: choices/options list
    if options is None:
        choices = _first_present(ex, ["choices", "options", "candidates"])
        if isinstance(choices, list) and len(choices) >= 4:
            options = [str(choices[i]) for i in range(4)]

    # schema 3: one correct + three incorrect
    if options is None:
        correct = _first_present(ex, ["Correct Answer", "correct_answer", "answer", "gold"])
        i1 = _first_present(ex, ["Incorrect Answer 1", "incorrect_answer_1", "distractor1"])
        i2 = _first_present(ex, ["Incorrect Answer 2", "incorrect_answer_2", "distractor2"])
        i3 = _first_present(ex, ["Incorrect Answer 3", "incorrect_answer_3", "distractor3"])
        if correct is not None and i1 is not None and i2 is not None and i3 is not None:
            pool = [str(correct), str(i1), str(i2), str(i3)]
            r = random.Random(2024 + idx)  # deterministic shuffle
            r.shuffle(pool)
            options = pool

    if options is None or len(options) < 4:
        raise KeyError(f"GPQA-D idx={idx} cannot build 4 options, keys={list(ex.keys())}")

    gt_letter = None
    ans = _first_present(
        ex,
        ["label", "answer", "correct", "correct_option", "answer_letter", "target", "gold_label"],
    )
    if ans is not None:
        ans_s = str(ans).strip()
        if ans_s in {"A", "B", "C", "D"}:
            gt_letter = ans_s
        elif ans_s in {"0", "1", "2", "3"}:
            gt_letter = ["A", "B", "C", "D"][int(ans_s)]
        else:
            for i, opt in enumerate(options[:4]):
                if str(opt).strip() == ans_s:
                    gt_letter = ["A", "B", "C", "D"][i]
                    break

    if gt_letter is None:
        corr_text = _first_present(ex, ["Correct Answer", "correct_answer", "gold"])
        if corr_text is not None:
            corr_s = str(corr_text).strip()
            for i, opt in enumerate(options[:4]):
                if str(opt).strip() == corr_s:
                    gt_letter = ["A", "B", "C", "D"][i]
                    break

    if gt_letter is None:
        raise KeyError(f"GPQA-D idx={idx} cannot infer gold option, keys={list(ex.keys())}")

    return str(question).strip(), [str(x).strip() for x in options[:4]], gt_letter


def _load_gpqa_d_local_csv(csv_path: str) -> list[dict[str, Any]]:
    p = os.path.expanduser(csv_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"GPQA 本地 CSV 不存在: {p}")
    rows: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"GPQA 本地 CSV 为空: {p}")
    return rows


def iter_gpqa_d_test(
    gpqa_d_local_csv: str | None = None,
    gpqa_d_dataset: str | None = None,
    gpqa_d_config: str | None = None,
    gpqa_d_split: str | None = None,
):
    if gpqa_d_local_csv:
        ds_local = _load_gpqa_d_local_csv(gpqa_d_local_csv)
        iterator = enumerate(ds_local)
    else:
        ds = _load_gpqa_d_hf(
            dataset_name=gpqa_d_dataset,
            dataset_config=gpqa_d_config,
            dataset_split=gpqa_d_split,
        )
        iterator = ((idx, ds[idx]) for idx in range(len(ds)))

    for idx, ex in iterator:
        q, choices, gt_letter = _extract_gpqa_d_row(ex, idx)
        yield _row(
            _user_content_gpqa_d(q, choices),
            gt_letter,
            "gpqa_d",
            extra_info={
                "index": idx,
                "split": "test",
                "source": "gpqa_diamond_csv" if gpqa_d_local_csv else "gpqa_diamond",
                "subject": ex.get("subject"),
            },
            ability="science",
        )


def _write_parquet(path: str, gen):
    ds = Dataset.from_generator(gen)
    ds.to_parquet(path)
    return len(ds)


def _format_exception_chain(e: BaseException) -> str:
    parts: list[str] = []
    cur: BaseException | None = e
    depth = 0
    while cur is not None and depth < 8:
        parts.append(f"{type(cur).__name__}: {cur}")
        nxt = cur.__cause__ if cur.__cause__ is not None else cur.__context__
        if nxt is cur:
            break
        cur = nxt
        depth += 1
    return " <- ".join(parts)


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
    ap.add_argument(
        "--include_gpqa_d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否尝试下载并导出 GPQA-D（diamond）测试集。",
    )
    ap.add_argument(
        "--gpqa_d_required",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="若为 true，GPQA-D 下载/处理失败时直接报错退出；否则仅告警并跳过。",
    )
    ap.add_argument(
        "--gpqa_d_dataset",
        type=str,
        default=None,
        help="可选，手动指定 GPQA 数据集名（如 Idavidrein/gpqa）。",
    )
    ap.add_argument(
        "--gpqa_d_local_csv",
        type=str,
        default=None,
        help="优先使用本地 GPQA-D CSV 文件路径（例如 gpqa_diamond.csv），不走 HF 下载。",
    )
    ap.add_argument(
        "--gpqa_d_config",
        type=str,
        default=None,
        help="可选，手动指定 GPQA 配置名（如 diamond 或 gpqa_diamond）。",
    )
    ap.add_argument(
        "--gpqa_d_split",
        type=str,
        default="test",
        help="GPQA 使用的 split，默认 test。",
    )
    ap.add_argument(
        "--include_extra_math_tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否额外导出 GSM8K(test) 与 MATH-lighteval(test)。",
    )
    ap.add_argument(
        "--include_math_hard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否导出 math_hard_test（从 MATH-lighteval 按 level 过滤）。",
    )
    ap.add_argument(
        "--math_hard_min_level",
        type=int,
        default=5,
        help="math_hard 过滤的最小 level（默认 5）。",
    )
    ap.add_argument(
        "--include_amc23",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否导出 AMC 2023 测试集（math-ai/amc23）。",
    )
    ap.add_argument(
        "--include_olympiadbench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否导出 OlympiadBench 测试集（math-ai/olympiadbench）。",
    )
    ap.add_argument(
        "--include_minerva_math",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否导出 Minerva-Math 测试集（默认 svc-huggingface/minerva-math）。",
    )
    ap.add_argument(
        "--minerva_math_dataset",
        type=str,
        default="svc-huggingface/minerva-math",
        help="Minerva-Math 的 HuggingFace 数据集名（需含 problem/solution 列，与官方 Minerva-Math 一致）。",
    )
    args = ap.parse_args()
    out = os.path.expanduser(args.output_dir)
    os.makedirs(out, exist_ok=True)

    if args.train_source == "processed":
        f_train = os.path.join(out, "dapo_math_17k_processed_train.parquet")

        def gen_train():
            yield from iter_dapo_processed_train(args.max_train_samples)

        print("1/4 open-r1/DAPO-Math-17k-Processed (train) ...")
        n1 = _write_parquet(f_train, gen_train)
    else:
        f_train = os.path.join(out, "dapo_math_byted_train.parquet")

        def gen_byted():
            yield from iter_byted_dapo_train(args.max_train_samples)

        print("1/4 BytedTsinghua-SIA/DAPO-Math-17k (train, 大规模) ...")
        n1 = _write_parquet(f_train, gen_byted)

    print(f"    -> {n1} rows, {f_train}")

    f_m500 = os.path.join(out, "math500_test.parquet")
    f_aime = os.path.join(out, "aime2024_test.parquet")
    f_aime25 = os.path.join(out, "aime2025_test.parquet")
    f_gsm8k = os.path.join(out, "gsm8k_test.parquet")
    f_math_lighteval = os.path.join(out, "math_lighteval_test.parquet")
    f_math_hard = os.path.join(out, "math_hard_test.parquet")
    f_gpqa = os.path.join(out, "gpqa_d_test.parquet")

    print("2/4 MATH-500 (test, DAPO-Processed 同款 prompt) ...")
    n2 = _write_parquet(f_m500, iter_math500_test)
    print(f"    -> {n2} rows, {f_m500}")

    print("3/4 AIME 2024 (test, 同上) ...")
    n3 = _write_parquet(f_aime, iter_aime24_test)
    print(f"    -> {n3} rows, {f_aime}")

    print("4/5 AIME 2025 (test, 同上) ...")
    n3b = _write_parquet(f_aime25, iter_aime25_test)
    print(f"    -> {n3b} rows, {f_aime25}")

    val_files = [f_m500, f_aime, f_aime25]

    f_amc23 = os.path.join(out, "amc23_test.parquet")
    f_olymp = os.path.join(out, "olympiadbench_test.parquet")
    f_minerva = os.path.join(out, "minerva_math_test.parquet")

    if bool(args.include_amc23):
        print("5/8 AMC 2023 (math-ai/amc23, test) ...")
        n_amc = _write_parquet(f_amc23, iter_amc23_test)
        print(f"    -> {n_amc} rows, {f_amc23}")
        val_files.append(f_amc23)
    else:
        print("5/8 AMC 2023 (math-ai/amc23, test) ... skipped (--no-include_amc23)")

    if bool(args.include_olympiadbench):
        print("6/8 OlympiadBench (math-ai/olympiadbench, test) ...")
        n_ob = _write_parquet(f_olymp, iter_olympiadbench_test)
        print(f"    -> {n_ob} rows, {f_olymp}")
        val_files.append(f_olymp)
    else:
        print("6/8 OlympiadBench (math-ai/olympiadbench, test) ... skipped (--no-include_olympiadbench)")

    if bool(args.include_minerva_math):
        print(f"7/8 Minerva-Math ({args.minerva_math_dataset}, test) ...")

        def gen_minerva():
            yield from iter_minerva_math_test(dataset_id=str(args.minerva_math_dataset).strip())

        n_mv = _write_parquet(f_minerva, gen_minerva)
        print(f"    -> {n_mv} rows, {f_minerva}")
        val_files.append(f_minerva)
    else:
        print("7/8 Minerva-Math ... skipped (--no-include_minerva_math)")

    if bool(args.include_extra_math_tests):
        print("8/9 GSM8K (test, 同款 prompt) ...")
        n4 = _write_parquet(f_gsm8k, iter_gsm8k_test)
        print(f"    -> {n4} rows, {f_gsm8k}")
        val_files.append(f_gsm8k)

        print("9/10 MATH-lighteval (test, 同款 prompt) ...")
        n5 = _write_parquet(f_math_lighteval, iter_math_lighteval_test)
        print(f"    -> {n5} rows, {f_math_lighteval}")
        val_files.append(f_math_lighteval)
    else:
        print("8/9 GSM8K (test, 同款 prompt) ... skipped (--no-include_extra_math_tests)")
        print("9/10 MATH-lighteval (test, 同款 prompt) ... skipped (--no-include_extra_math_tests)")

    if bool(args.include_math_hard):
        print("10/11 math_hard (MATH-lighteval level filtered) ...")

        def gen_math_hard():
            yield from iter_math_hard_test(min_level=int(args.math_hard_min_level))

        n_hard = _write_parquet(f_math_hard, gen_math_hard)
        print(f"    -> {n_hard} rows, {f_math_hard}")
        val_files.append(f_math_hard)
    else:
        print("10/11 math_hard (MATH-lighteval level filtered) ... skipped (--no-include_math_hard)")

    gpqa_enabled = bool(args.include_gpqa_d)
    gpqa_written = False
    if gpqa_enabled:
        print("11/11 GPQA-D (diamond test, multiple-choice) ...")

        try:
            # Pre-materialize to surface the real underlying exception instead of
            # a generic DatasetGenerationError wrapper from Dataset.from_generator.
            gpqa_rows = list(
                iter_gpqa_d_test(
                    gpqa_d_local_csv=args.gpqa_d_local_csv,
                    gpqa_d_dataset=args.gpqa_d_dataset,
                    gpqa_d_config=args.gpqa_d_config,
                    gpqa_d_split=args.gpqa_d_split,
                )
            )
            ds_gpqa = Dataset.from_list(gpqa_rows)
            ds_gpqa.to_parquet(f_gpqa)
            n4 = len(ds_gpqa)
            gpqa_written = True
            print(f"    -> {n4} rows, {f_gpqa}")
        except Exception as e:
            if bool(args.gpqa_d_required):
                raise
            print(f"    !! 跳过 GPQA-D：{_format_exception_chain(e)}")
    else:
        print("11/11 GPQA-D (diamond test, multiple-choice) ... skipped (--no-include_gpqa_d)")

    print("\n训练 / 验证可设为:")
    print(f'  data.train_files="{f_train}"')
    if gpqa_written:
        val_files.append(f_gpqa)
    val_files_str = ",".join([f"'{p}'" for p in val_files])
    print(f'  data.val_files="[{val_files_str}]"')


if __name__ == "__main__":
    main()
