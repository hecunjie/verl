#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
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
    file_sync,
    generate_rollout_vllm,
    init_dist,
    load_data,
    purge_all_torchrun_like_env_for_vllm_standalone,
)


def parse_int_list(text: str) -> list[int]:
    vals = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("m_grid must contain at least one positive integer")
    return vals


def pick_positions_by_entropy(entropies: list[float], max_positions: int) -> list[int]:
    if not entropies:
        return []
    order = np.argsort(np.array(entropies))
    k = min(max_positions, len(entropies))
    return sorted(int(i) for i in order[-k:].tolist())


def estimate_suffix_return_samples(
    llm: Any,
    prefix_ids: list[int],
    m_max: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logprobs_k: int,
) -> list[float]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from entropy_credit_experiment import vllm_generate_quiet

    returns: list[float] = []
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=clamp_vllm_logprobs_topk(logprobs_k),
    )
    for _ in range(m_max):
        out = vllm_generate_quiet(llm, [TokensPrompt(prompt_token_ids=prefix_ids)], sp)
        o = out[0].outputs[0]
        step_logprobs = []
        for step_lp in o.logprobs or []:
            d = {int(tid): float(info.logprob) for tid, info in step_lp.items()}
            step_logprobs.append(d)
        entropies = [entropy_from_logprobs_topk(s) for s in step_logprobs]
        returns.append(float(sum(entropies)))
    return returns


def summarize_variance_from_samples(samples: list[float], m_grid: list[int]) -> dict[str, Any]:
    arr = np.array(samples, dtype=np.float64)
    if arr.size < 2:
        # Degenerate case; keep shape stable.
        out = {}
        for m in m_grid:
            out[str(m)] = {"var_estimator": 0.0, "cv_estimator": 0.0, "ci95_width": 0.0}
        return out
    s2 = float(np.var(arr, ddof=1))
    mu = float(np.mean(arr))
    out = {}
    for m in m_grid:
        var_est = s2 / float(m)
        std_est = math.sqrt(max(var_est, 0.0))
        cv = std_est / (abs(mu) + 1e-8)
        ci95_width = 2.0 * 1.96 * std_est
        out[str(m)] = {
            "var_estimator": float(var_est),
            "cv_estimator": float(cv),
            "ci95_width": float(ci95_width),
        }
    return out


def pick_recommended_m(stats_by_m: dict[str, dict[str, float]], m_grid: list[int], ci95_threshold: float) -> int:
    # Prefer smallest m meeting CV and CI constraints; else return max.
    for m in m_grid:
        st = stats_by_m[str(m)]
        if st["cv_estimator"] < 0.10 and st["ci95_width"] < ci95_threshold:
            return m
    return m_grid[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate MC sample count M for Method2 F_t estimation variance.")
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--rollouts_per_prompt", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--vllm_logprobs_topk", type=int, default=20)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--m_grid", type=str, default="2,4,8,12,16")
    parser.add_argument("--max_positions_per_rollout", type=int, default=10)
    parser.add_argument("--ci95_threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm_shard_rank", type=int, default=None)
    parser.add_argument("--vllm_shard_world_size", type=int, default=None)
    args = parser.parse_args()

    m_grid = parse_int_list(args.m_grid)
    m_max = m_grid[-1]

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
    part_path = out_dir / f"calibrate_m_rank{rank}.jsonl"
    with open(part_path, "w", encoding="utf-8"):
        pass

    rows = load_data(args.input_data, args.max_samples, args.seed)
    local_rows = [row for idx, row in enumerate(rows) if idx % world_size == rank]

    for local_i, row in enumerate(local_rows):
        global_idx = local_i * world_size + rank
        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

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
            entropies = [entropy_from_logprobs_topk(s) for s in scores[: len(response_ids)]]
            positions = pick_positions_by_entropy(entropies, args.max_positions_per_rollout)

            for pos in positions:
                # Prefix before token at index pos.
                prefix_ids = prompt_ids + response_ids[:pos]
                suffix_returns = estimate_suffix_return_samples(
                    llm=llm,
                    prefix_ids=prefix_ids,
                    m_max=m_max,
                    max_new_tokens=max(1, args.max_new_tokens - pos),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    logprobs_k=args.vllm_logprobs_topk,
                )
                stats_by_m = summarize_variance_from_samples(suffix_returns, m_grid)
                rec = {
                    "sample_index": global_idx,
                    "rollout_index": rollout_idx,
                    "token_index": pos,
                    "entropy_t": float(entropies[pos]),
                    "m_grid_stats": stats_by_m,
                    "recommended_m_local": pick_recommended_m(stats_by_m, m_grid, args.ci95_threshold),
                }
                with open(part_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    file_sync(out_dir=out_dir, rank=rank, world_size=world_size, tag="done_calib")

    if rank == 0:
        merged: list[dict[str, Any]] = []
        for r in range(world_size):
            p = out_dir / f"calibrate_m_rank{r}.jsonl"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged.append(json.loads(line))
        with open(out_dir / "calibrate_m_merged.jsonl", "w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if merged:
            summary: dict[str, Any] = {"count_prefixes": len(merged), "m_grid": m_grid}
            per_m_cv = {m: [] for m in m_grid}
            per_m_ci = {m: [] for m in m_grid}
            local_recs = []
            for rec in merged:
                local_recs.append(int(rec["recommended_m_local"]))
                for m in m_grid:
                    st = rec["m_grid_stats"][str(m)]
                    per_m_cv[m].append(float(st["cv_estimator"]))
                    per_m_ci[m].append(float(st["ci95_width"]))
            summary["median_cv_by_m"] = {str(m): float(np.median(per_m_cv[m])) for m in m_grid}
            summary["median_ci95_width_by_m"] = {str(m): float(np.median(per_m_ci[m])) for m in m_grid}
            summary["recommended_m_global"] = int(statistics.median(local_recs))
            with open(out_dir / "calibrate_m_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

