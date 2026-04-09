#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from entropy_credit_experiment import (
    _configure_vllm_ipc_for_single_node,
    _configure_vllm_multiprocessing_spawn,
    _snapshot_and_clear_torchrun_dist_env,
    _restore_torchrun_dist_env,
    build_prompt_text,
    clamp_vllm_logprobs_topk,
    entropy_from_logprobs_topk,
    evaluate_solution_acc,
    file_sync,
    generate_rollout_vllm,
    init_dist,
    load_data,
    purge_all_torchrun_like_env_for_vllm_standalone,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def pick_top_entropy_positions(entropies: list[float], top_ratio: float, max_positions: int) -> list[int]:
    if not entropies:
        return []
    k = max(1, int(math.ceil(len(entropies) * top_ratio)))
    if max_positions > 0:
        k = min(k, max_positions)
    order = np.argsort(np.array(entropies))
    return sorted(int(i) for i in order[-k:].tolist())


def estimate_F_from_prefix_vllm(
    llm: Any,
    prefix_ids: list[int],
    m_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logprobs_k: int,
) -> float:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from entropy_credit_experiment import vllm_generate_quiet

    vals = []
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=clamp_vllm_logprobs_topk(logprobs_k),
    )
    for _ in range(m_samples):
        out = vllm_generate_quiet(llm, [TokensPrompt(prompt_token_ids=prefix_ids)], sp)
        o = out[0].outputs[0]
        step_logprobs = []
        for step_lp in o.logprobs or []:
            d = {int(tid): float(info.logprob) for tid, info in step_lp.items()}
            step_logprobs.append(d)
        entropies = [entropy_from_logprobs_topk(s) for s in step_logprobs]
        vals.append(float(sum(entropies)))
    return float(np.mean(vals)) if vals else 0.0


def topk_candidates_with_probs(step_logprobs: dict[int, float], k: int) -> tuple[list[int], list[float]]:
    if not step_logprobs:
        return [], []
    items = sorted(step_logprobs.items(), key=lambda x: x[1], reverse=True)[:k]
    tids = [int(t) for t, _ in items]
    lps = np.array([float(v) for _, v in items], dtype=np.float64)
    m = float(np.max(lps))
    p = np.exp(np.clip(lps - m, -80.0, 0.0))
    z = float(np.sum(p))
    if z <= 0 or not np.isfinite(z):
        probs = np.ones(len(tids), dtype=np.float64) / float(max(1, len(tids)))
    else:
        probs = p / z
    return tids, [float(x) for x in probs.tolist()]


def compute_group_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Method2 bias term on paired correct/wrong rollouts.")
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--rollouts_per_prompt", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--vllm_logprobs_topk", type=int, default=20)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--top_entropy_ratio", type=float, default=0.10)
    parser.add_argument("--max_positions_per_rollout", type=int, default=20)
    parser.add_argument("--mc_m_samples", type=int, default=4)
    parser.add_argument("--topk_alt", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm_shard_rank", type=int, default=None)
    parser.add_argument("--vllm_shard_world_size", type=int, default=None)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--progress_all_ranks",
        action="store_true",
        help="Show tqdm on every shard process (default: rank 0 only).",
    )
    args = parser.parse_args()

    vllm_standalone = args.vllm_shard_rank is not None and args.vllm_shard_world_size is not None
    if (args.vllm_shard_rank is None) ^ (args.vllm_shard_world_size is None):
        raise SystemExit("Pass both --vllm_shard_rank and --vllm_shard_world_size, or neither.")
    if vllm_standalone:
        purge_all_torchrun_like_env_for_vllm_standalone()

    _configure_vllm_multiprocessing_spawn()
    _configure_vllm_ipc_for_single_node()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dist_snap = _snapshot_and_clear_torchrun_dist_env()
    try:
        from vllm import LLM

        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
            max_model_len=int(args.vllm_max_model_len),
            enforce_eager=True,
        )
    finally:
        _restore_torchrun_dist_env(dist_snap)

    _, rank, world_size = init_dist(
        backend="vllm",
        rank_override=args.vllm_shard_rank if vllm_standalone else None,
        world_size_override=args.vllm_shard_world_size if vllm_standalone else None,
    )

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"pair_bias_rank{rank}.jsonl"
    with open(part_path, "w", encoding="utf-8"):
        pass

    rows = load_data(args.input_data, args.max_samples, args.seed)
    local_rows = [row for idx, row in enumerate(rows) if idx % world_size == rank]

    env_no_tqdm = os.environ.get("TQDM_DISABLE", "").strip().lower() in ("1", "true", "yes")
    use_tqdm = tqdm is not None and not args.no_progress and not env_no_tqdm and (
        args.progress_all_ranks or rank == 0
    )
    bar_pos = int(rank) if use_tqdm else 0
    shard_desc = f"shard{rank}"

    prompt_iter = enumerate(local_rows)
    if use_tqdm:
        prompt_iter = tqdm(
            prompt_iter,
            total=len(local_rows),
            desc=f"{shard_desc} prompts",
            dynamic_ncols=True,
            position=bar_pos,
            leave=True,
            mininterval=0.5,
        )

    for local_i, row in prompt_iter:
        global_idx = local_i * world_size + rank
        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        data_source = row.get("data_source", "math_dapo")
        ground_truth = str((row.get("reward_model") or {}).get("ground_truth", ""))

        rollouts: list[dict[str, Any]] = []
        for rollout_idx in range(args.rollouts_per_prompt):
            response_ids, scores = generate_rollout_vllm(
                llm=llm,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                logprobs_k=args.vllm_logprobs_topk,
            )
            if not response_ids:
                continue
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            acc, _ = evaluate_solution_acc(data_source=data_source, solution_str=response_text, ground_truth=ground_truth)
            entropies = [entropy_from_logprobs_topk(s) for s in scores[: len(response_ids)]]
            rollouts.append(
                {
                    "rollout_index": rollout_idx,
                    "response_ids": response_ids,
                    "scores": scores,
                    "entropies": entropies,
                    "is_correct": bool(acc),
                }
            )

        correct = [r for r in rollouts if r["is_correct"]]
        wrong = [r for r in rollouts if not r["is_correct"]]
        if not correct or not wrong:
            continue

        # Deterministic selection: median-length candidate per group.
        def select_median_length(items: list[dict[str, Any]]) -> dict[str, Any]:
            items_sorted = sorted(items, key=lambda x: len(x["response_ids"]))
            return items_sorted[len(items_sorted) // 2]

        pair = [("correct", select_median_length(correct)), ("wrong", select_median_length(wrong))]
        for group_name, rr in pair:
            entropies = rr["entropies"]
            response_ids = rr["response_ids"]
            scores = rr["scores"]
            positions = pick_top_entropy_positions(entropies, args.top_entropy_ratio, args.max_positions_per_rollout)
            for pos in positions:
                if pos >= len(scores):
                    continue
                step_lp = scores[pos]
                tids, probs = topk_candidates_with_probs(step_lp, args.topk_alt)
                if not tids:
                    continue

                # F(x_{<=t}) for actually selected token:
                prefix_selected = prompt_ids + response_ids[: pos + 1]
                f_selected = estimate_F_from_prefix_vllm(
                    llm=llm,
                    prefix_ids=prefix_selected,
                    m_samples=args.mc_m_samples,
                    max_new_tokens=max(1, args.max_new_tokens - pos - 1),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    logprobs_k=args.vllm_logprobs_topk,
                )

                # \bar F_t approx with top-k alternatives
                f_bar = 0.0
                for tid, p in zip(tids, probs, strict=False):
                    prefix_alt = prompt_ids + response_ids[:pos] + [int(tid)]
                    f_alt = estimate_F_from_prefix_vllm(
                        llm=llm,
                        prefix_ids=prefix_alt,
                        m_samples=args.mc_m_samples,
                        max_new_tokens=max(1, args.max_new_tokens - pos - 1),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        logprobs_k=args.vllm_logprobs_topk,
                    )
                    f_bar += float(p) * float(f_alt)

                h_t = float(entropies[pos])
                bias_t = float(f_bar - f_selected)
                delta_hat = float(h_t + bias_t)

                rec = {
                    "sample_index": global_idx,
                    "group": group_name,
                    "rollout_index": rr["rollout_index"],
                    "token_index": int(pos),
                    "entropy_t": h_t,
                    "bias_t": bias_t,
                    "delta_hat_t": delta_hat,
                    "bias_over_entropy": float(bias_t / (h_t + 1e-8)),
                    "selected_token_id": int(response_ids[pos]),
                    "topk_alt_token_ids": [int(t) for t in tids],
                    "topk_alt_probs": [float(p) for p in probs],
                }
                with open(part_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    file_sync(out_dir=out_dir, rank=rank, world_size=world_size, tag="done_pair")

    if rank == 0:
        merged: list[dict[str, Any]] = []
        for r in range(world_size):
            p = out_dir / f"pair_bias_rank{r}.jsonl"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged.append(json.loads(line))

        with open(out_dir / "pair_bias_merged.jsonl", "w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        corr_bias = [float(r["bias_t"]) for r in merged if r["group"] == "correct"]
        wrong_bias = [float(r["bias_t"]) for r in merged if r["group"] == "wrong"]
        corr_delta = [float(r["delta_hat_t"]) for r in merged if r["group"] == "correct"]
        wrong_delta = [float(r["delta_hat_t"]) for r in merged if r["group"] == "wrong"]

        summary = {
            "num_records": len(merged),
            "correct_bias": compute_group_stats(corr_bias),
            "wrong_bias": compute_group_stats(wrong_bias),
            "correct_delta_hat": compute_group_stats(corr_delta),
            "wrong_delta_hat": compute_group_stats(wrong_delta),
            "p_bias_positive_correct": float(np.mean(np.array(corr_bias) > 0)) if corr_bias else float("nan"),
            "p_bias_positive_wrong": float(np.mean(np.array(wrong_bias) > 0)) if wrong_bias else float("nan"),
        }
        with open(out_dir / "pair_bias_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

