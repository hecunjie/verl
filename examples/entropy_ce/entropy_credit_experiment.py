#!/usr/bin/env python3
"""
Delta-H entropy credit experiment for VERL-format math data.

Features:
- Multi-GPU analysis via torchrun.
- Single model path input.
- VERL-format parquet/json input support.
- max_samples support.
- Phase2 defaults to Method B (outcome flip rate).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score import default_compute_score


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def barrier() -> None:
    if is_dist():
        dist.barrier()


def init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def load_data(path: str, max_samples: int, seed: int) -> list[dict[str, Any]]:
    if path.endswith(".parquet"):
        ds = datasets.load_dataset("parquet", data_files=path)["train"]
    elif path.endswith(".json") or path.endswith(".jsonl"):
        ds = datasets.load_dataset("json", data_files=path)["train"]
    else:
        raise ValueError(f"Unsupported input format: {path}")

    total = len(ds)
    if 0 < max_samples < total:
        indices = list(range(total))
        random.Random(seed).shuffle(indices)
        ds = ds.select(indices[:max_samples])
    return [ds[i] for i in range(len(ds))]


def build_prompt_text(tokenizer, prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, list):
        return tokenizer.apply_chat_template(prompt_obj, add_generation_prompt=True, tokenize=False)
    raise ValueError(f"Unsupported prompt type: {type(prompt_obj)}")


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-20))
    return float((-probs * log_probs).sum().item())


def suffix_avg(values: list[float], start: int) -> float:
    if start >= len(values):
        return 0.0
    tail = values[start:]
    return float(sum(tail) / len(tail))


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)
    xr = rankdata_average_ties(xa)
    yr = rankdata_average_ties(ya)
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    denom = math.sqrt(float((xc * xc).sum() * (yc * yc).sum()))
    if denom == 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def precision_at_k(signal: list[float], importance: list[float], k: int) -> float:
    if not signal or not importance:
        return float("nan")
    k = min(k, len(signal), len(importance))
    if k <= 0:
        return float("nan")
    sig_idx = np.argsort(np.array(signal))[-k:]
    imp_idx = set(np.argsort(np.array(importance))[-k:].tolist())
    hit = sum(int(i in imp_idx) for i in sig_idx.tolist())
    return float(hit / k)


@dataclass
class RolloutItem:
    sample_index: int
    rollout_index: int
    data_source: str
    ground_truth: str
    prompt_text: str
    response_text: str
    response_token_ids: list[int]
    entropies: list[float]
    deltas: list[float]
    varentropies: list[float]
    branching_factor: list[float]
    importance_method_b: list[float]


def compute_varentropy(logits: torch.Tensor, entropy: float) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-20))
    center = log_probs + entropy
    return float((probs * (center * center)).sum().item())


def generate_rollout(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[int], list[torch.Tensor]]:
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    attn = encoded["attention_mask"].to(model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out.sequences[0, input_ids.shape[1] :].tolist()
    scores = [s[0].detach().float().cpu() for s in out.scores]
    return gen_ids, scores


def method_b_importance(
    model,
    tokenizer,
    prompt_text: str,
    response_ids: list[int],
    scores: list[torch.Tensor],
    data_source: str,
    ground_truth: str,
    selected_positions: list[int],
    m_samples: int,
    top_k_alt: int,
) -> list[float]:
    base_response = tokenizer.decode(response_ids, skip_special_tokens=True)
    base_result = default_compute_score(data_source=data_source, solution_str=base_response, ground_truth=ground_truth)
    base_acc = bool(base_result["acc"] if isinstance(base_result, dict) and "acc" in base_result else float(base_result) > 0.5)

    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].tolist()
    importance = [0.0 for _ in range(len(response_ids))]

    for pos in selected_positions:
        if pos < 0 or pos >= len(response_ids) or pos >= len(scores):
            continue
        logits = scores[pos]
        tk = min(max(2, top_k_alt), logits.shape[-1])
        topk = torch.topk(logits, k=tk, dim=-1)
        candidates = topk.indices.tolist()
        probs = torch.softmax(topk.values, dim=-1).numpy()
        alt_tokens = [c for c in candidates if c != response_ids[pos]]
        if not alt_tokens:
            continue
        alt_probs = np.array([probs[candidates.index(t)] for t in alt_tokens], dtype=np.float64)
        alt_probs = alt_probs / alt_probs.sum()

        flips = 0
        for _ in range(m_samples):
            sampled = int(np.random.choice(np.array(alt_tokens), p=alt_probs))
            mutated = response_ids[:]
            mutated[pos] = sampled

            prefix = prompt_ids + mutated[: pos + 1]
            rem_len = max(1, len(response_ids) - pos - 1)
            prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=model.device).unsqueeze(0)
            attn = torch.ones_like(prefix_tensor)
            out = model.generate(
                input_ids=prefix_tensor,
                attention_mask=attn,
                max_new_tokens=rem_len,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = out.sequences[0, prefix_tensor.shape[1] :].tolist()
            full_mutated = mutated[: pos + 1] + new_ids
            new_resp = tokenizer.decode(full_mutated, skip_special_tokens=True)
            res = default_compute_score(data_source=data_source, solution_str=new_resp, ground_truth=ground_truth)
            acc = bool(res["acc"] if isinstance(res, dict) and "acc" in res else float(res) > 0.5)
            if acc != base_acc:
                flips += 1
        importance[pos] = float(flips / m_samples)
    return importance


def pick_candidate_positions(deltas: list[float], entropies: list[float], seed: int) -> list[int]:
    n = len(entropies)
    if n == 0:
        return []
    k = max(1, int(math.ceil(n * 0.1)))
    idx_delta = np.argsort(np.array(deltas))[-k:].tolist()
    idx_entropy = np.argsort(np.array(entropies))[-k:].tolist()
    rng = random.Random(seed)
    all_idx = list(range(n))
    rng.shuffle(all_idx)
    idx_random = all_idx[:k]
    return sorted(set(idx_delta + idx_entropy + idx_random))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="VERL format parquet/json/jsonl.")
    parser.add_argument("--model_path", type=str, required=True, help="Single model path.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--rollouts_per_prompt", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--phase2_method", type=str, default="B", choices=["A", "B"])
    parser.add_argument("--method_b_m_samples", type=int, default=4)
    parser.add_argument("--method_b_topk_alt", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    local_rank, rank, world_size = init_dist()
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    part_file = out_dir / f"phase1_3_rank{rank}.jsonl"
    with open(part_file, "w", encoding="utf-8"):
        pass

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": local_rank} if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    rows = load_data(args.input_data, args.max_samples, args.seed)
    local_rows = [row for idx, row in enumerate(rows) if idx % world_size == rank]

    for local_i, row in enumerate(local_rows):
        global_idx = local_i * world_size + rank
        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        data_source = row.get("data_source", "math_dapo")
        ground_truth = str((row.get("reward_model") or {}).get("ground_truth", ""))

        for rollout_idx in range(args.rollouts_per_prompt):
            response_ids, scores = generate_rollout(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            if not response_ids:
                continue
            entropies = [entropy_from_logits(s) for s in scores[: len(response_ids)]]
            varentropies = [compute_varentropy(s, h) for s, h in zip(scores[: len(response_ids)], entropies, strict=False)]
            branching = [math.exp(h) for h in entropies]
            deltas = []
            for t in range(len(entropies)):
                et = suffix_avg(entropies, t + 1)
                et1 = suffix_avg(entropies, t + 2)
                deltas.append(et - et1)

            if args.phase2_method == "B":
                candidates = pick_candidate_positions(deltas, entropies, args.seed + global_idx + rollout_idx)
                importance = method_b_importance(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    response_ids=response_ids,
                    scores=scores,
                    data_source=data_source,
                    ground_truth=ground_truth,
                    selected_positions=candidates,
                    m_samples=args.method_b_m_samples,
                    top_k_alt=args.method_b_topk_alt,
                )
            else:
                importance = [0.0 for _ in range(len(response_ids))]

            item = RolloutItem(
                sample_index=global_idx,
                rollout_index=rollout_idx,
                data_source=data_source,
                ground_truth=ground_truth,
                prompt_text=prompt_text,
                response_text=tokenizer.decode(response_ids, skip_special_tokens=True),
                response_token_ids=response_ids,
                entropies=entropies,
                deltas=deltas,
                varentropies=varentropies,
                branching_factor=branching,
                importance_method_b=importance,
            )
            with open(part_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

    barrier()

    if rank == 0:
        all_records = []
        for r in range(world_size):
            fpath = out_dir / f"phase1_3_rank{r}.jsonl"
            if not fpath.exists():
                continue
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))
        merged_path = out_dir / "phase1_3_merged.jsonl"
        with open(merged_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        corr_delta, corr_h, corr_v = [], [], []
        p5_delta, p10_delta, p20_delta = [], [], []
        p5_h, p10_h, p20_h = [], [], []
        for rec in all_records:
            imp = rec["importance_method_b"]
            if not any(x > 0 for x in imp):
                continue
            delta = rec["deltas"]
            h = rec["entropies"]
            v = rec["varentropies"]
            corr_delta.append(spearman(delta, imp))
            corr_h.append(spearman(h, imp))
            corr_v.append(spearman(v, imp))
            p5_delta.append(precision_at_k(delta, imp, 5))
            p10_delta.append(precision_at_k(delta, imp, 10))
            p20_delta.append(precision_at_k(delta, imp, 20))
            p5_h.append(precision_at_k(h, imp, 5))
            p10_h.append(precision_at_k(h, imp, 10))
            p20_h.append(precision_at_k(h, imp, 20))

        def mean_no_nan(x: list[float]) -> float:
            arr = np.array(x, dtype=np.float64)
            if arr.size == 0:
                return float("nan")
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return float("nan")
            return float(arr.mean())

        metrics = {
            "num_rollouts": len(all_records),
            "phase2_method": args.phase2_method,
            "spearman_delta_vs_importance": mean_no_nan(corr_delta),
            "spearman_entropy_vs_importance": mean_no_nan(corr_h),
            "spearman_varentropy_vs_importance": mean_no_nan(corr_v),
            "precision@5_delta": mean_no_nan(p5_delta),
            "precision@10_delta": mean_no_nan(p10_delta),
            "precision@20_delta": mean_no_nan(p20_delta),
            "precision@5_entropy": mean_no_nan(p5_h),
            "precision@10_entropy": mean_no_nan(p10_h),
            "precision@20_entropy": mean_no_nan(p20_h),
        }
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
