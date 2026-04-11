#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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
    build_token_context,
    entropy_from_logprobs_topk,
    estimate_F_mc_many_prefixes_vllm,
    evaluate_solution_acc,
    file_sync,
    generate_rollouts_vllm_batched,
    init_dist,
    load_data,
    purge_all_torchrun_like_env_for_vllm_standalone,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _ground_truth_from_row(row: dict[str, Any]) -> str:
    """VERL parquet usually has ``reward_model.ground_truth`` (dict). GRPO exports may store
    ``reward_model`` as a JSON string or use ``label`` — empty GT makes every rollout look wrong.
    """
    rm = row.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        return str(rm["ground_truth"])
    if isinstance(rm, str) and rm.strip():
        try:
            obj = json.loads(rm)
            if isinstance(obj, dict) and obj.get("ground_truth") is not None:
                return str(obj["ground_truth"])
        except json.JSONDecodeError:
            pass
    if row.get("label") is not None:
        if isinstance(row["label"], list) and len(row["label"]) == 1:
            return str(row["label"][0])
        return str(row["label"])
    return ""


def pick_top_entropy_positions(
    entropies: list[float],
    scores: list[dict[int, float]],
    top_ratio: float,
    position_cap: int,
) -> list[int]:
    """Take ``k`` highest-entropy steps among indices with **non-empty** vLLM logprobs.

    ``k = min(position_cap, ceil(len(entropies) * top_ratio))`` capped by number of viable steps.

    Rationale: vLLM may return empty ``logprobs`` dicts on some steps (e.g. long generations). Those
    indices were still given entropy 0 in ``entropies``; taking ``argsort`` could mix them into the
    top-``k`` bucket, then ``topp`` sees ``{}`` and skips — often wiping the entire ``wrong`` branch
    while ``correct`` still writes lines.
    """
    if not entropies:
        return []
    n = min(len(entropies), len(scores))
    viable = [i for i in range(n) if scores[i]]
    if not viable:
        return []
    k_from_ratio = max(1, int(math.ceil(len(entropies) * float(top_ratio))))
    if position_cap > 0:
        k_budget = min(int(position_cap), k_from_ratio)
    else:
        k_budget = k_from_ratio
    k_use = min(k_budget, len(viable))
    viable_sorted = sorted(viable, key=lambda i: entropies[i])
    chosen = viable_sorted[-k_use:]
    return sorted(int(i) for i in chosen)


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


def topp_capped_candidates_with_probs(
    step_logprobs: dict[int, float],
    top_p: float,
    max_k: int,
) -> tuple[list[int], list[float], dict[str, int]]:
    """Smallest prefix covering ``top_p`` mass (on dict renormalized), then cap at ``max_k``.

    Mass is computed over **all** tokens returned in ``step_logprobs`` (vLLM truncates to
    ``logprobs_k``); if the tail is cut off, ``top_p`` is relative to that partial distribution only.
    """
    if not step_logprobs:
        return [], [], {"k_topp": 0, "k_used": 0}
    items = sorted(step_logprobs.items(), key=lambda x: x[1], reverse=True)
    tids_all = [int(t) for t, _ in items]
    lps = np.array([float(v) for _, v in items], dtype=np.float64)
    m = float(np.max(lps))
    p_full = np.exp(np.clip(lps - m, -80.0, 0.0))
    z = float(np.sum(p_full))
    if z <= 0 or not np.isfinite(z):
        p_full = np.ones(len(tids_all), dtype=np.float64) / float(len(tids_all))
    else:
        p_full = p_full / z
    cum = np.cumsum(p_full)
    tp = float(np.clip(top_p, 1e-12, 1.0))
    idx = int(np.searchsorted(cum, tp, side="left"))
    k_topp = min(len(p_full), max(1, idx + 1))
    k_used = min(int(max_k), k_topp)
    sel_p = p_full[:k_used].copy()
    z2 = float(np.sum(sel_p))
    if z2 <= 0 or not np.isfinite(z2):
        sel_p = np.ones(k_used, dtype=np.float64) / float(max(1, k_used))
    else:
        sel_p = sel_p / z2
    return tids_all[:k_used], [float(x) for x in sel_p.tolist()], {"k_topp": k_topp, "k_used": k_used}


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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Rollout / continuation length. R1-style models need room for CoT before Answer:\\boxed{}; "
        "too small → all rollouts truncate similarly → no mixed correct/wrong in 8 samples.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--vllm_logprobs_topk", type=int, default=20)
    parser.add_argument(
        "--vllm_request_batch_chunk",
        type=int,
        default=64,
        help="每次 llm.generate 最多并发的序列数（rollout 与 MC）。显存不够时调小。",
    )
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument(
        "--top_entropy_ratio",
        type=float,
        default=0.10,
        help="High-entropy position count uses ceil(ratio * response_length) before cap.",
    )
    parser.add_argument(
        "--max_positions_per_rollout",
        type=int,
        default=500,
        help="Upper cap: number of positions = min(this, ceil(top_entropy_ratio * seq_len)).",
    )
    parser.add_argument(
        "--context_window_tokens",
        type=int,
        default=64,
        help="Half-window (tokens) for context_left/right text around branch token in per-line records.",
    )
    parser.add_argument(
        "--per_sample_jsonl_subdir",
        type=str,
        default="per_sample",
        help="Under output_dir, one sample_{global_idx:06d}.jsonl per prompt (all positions/groups append).",
    )
    parser.add_argument(
        "--no_per_sample_jsonl",
        action="store_true",
        help="Do not write per-sample jsonl files (only pair_bias_rank*.jsonl).",
    )
    parser.add_argument(
        "--mc_m_samples",
        type=int,
        default=64,
        help="MC samples per F estimate (suffix entropy sum); higher = lower variance.",
    )
    parser.add_argument(
        "--f_continuation_mode",
        choices=["full", "first_sentence"],
        default="full",
        help="full: 一次 generate 续写至 max_new_tokens；first_sentence: 分块生成并在句末停（见 sentence_stop_utils，较慢）。",
    )
    parser.add_argument(
        "--f_sentence_chunk_max_tokens",
        type=int,
        default=48,
        help="f_continuation_mode=first_sentence 时每轮 vLLM.generate 的 max_tokens 上限。",
    )
    parser.add_argument(
        "--f_sentence_stop",
        choices=["simple", "pysbd"],
        default="simple",
        help="句末判定：simple=中英启发式+小数点过滤；pysbd=英文 pysbd（需 pip install pysbd）。",
    )
    parser.add_argument(
        "--candidate_mode",
        choices=["topp", "fixed"],
        default="topp",
        help="Alternatives for bar F_t: topp = min(candidate_max_k, smallest k with cum mass>=candidate_top_p) "
        "on renormalized vLLM logprobs dict; fixed = top-k by logprob with --topk_alt.",
    )
    parser.add_argument(
        "--candidate_top_p",
        type=float,
        default=0.9,
        help="Nucleus mass threshold within the returned logprobs dict (candidate_mode=topp).",
    )
    parser.add_argument(
        "--candidate_max_k",
        type=int,
        default=5,
        help="Max number of alternative tokens (candidate_mode=topp).",
    )
    parser.add_argument("--topk_alt", type=int, default=3, help="Top-k alternatives when candidate_mode=fixed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vllm_shard_rank",
        type=int,
        default=None,
        help="Data shard index in [0, world_size). Multi-node: global rank = NODE_RANK * local_gpu_count + gpu_index.",
    )
    parser.add_argument(
        "--vllm_shard_world_size",
        type=int,
        default=None,
        help="Total vLLM worker count across all nodes (e.g. NNODES * NPROC_PER_NODE).",
    )
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--progress_all_ranks",
        action="store_true",
        help="Show tqdm on every shard process (default: rank 0 only).",
    )
    parser.add_argument(
        "--progress_nested",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="嵌套 tqdm：rollout → 高熵位置 → 每次 F 估计的 MC（默认开启）。关闭：--no-progress_nested。",
    )
    parser.add_argument(
        "--progress_echo",
        action="store_true",
        help="Rank 0 only: stderr 每处理完一条 prompt 打一行耗时（有 tqdm 时也保留，便于估总时长）。",
    )
    args = parser.parse_args()
    if not (0.0 < float(args.candidate_top_p) <= 1.0):
        raise SystemExit("--candidate_top_p must be in (0, 1].")
    if int(args.candidate_max_k) < 1:
        raise SystemExit("--candidate_max_k must be >= 1.")
    if args.candidate_mode == "fixed" and int(args.topk_alt) < 1:
        raise SystemExit("--topk_alt must be >= 1 when candidate_mode=fixed.")
    if int(args.vllm_request_batch_chunk) < 1:
        raise SystemExit("--vllm_request_batch_chunk must be >= 1.")
    if int(args.max_positions_per_rollout) < 1:
        raise SystemExit("--max_positions_per_rollout must be >= 1 (use large value e.g. 500 for cap).")
    if int(args.context_window_tokens) < 0:
        raise SystemExit("--context_window_tokens must be >= 0.")
    if int(args.f_sentence_chunk_max_tokens) < 1:
        raise SystemExit("--f_sentence_chunk_max_tokens must be >= 1.")

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
    per_sample_dir: Path | None = None
    if not args.no_per_sample_jsonl:
        per_sample_dir = out_dir / str(args.per_sample_jsonl_subdir).strip().replace("..", "_")
        per_sample_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"pair_bias_rank{rank}.jsonl"
    with open(part_path, "w", encoding="utf-8"):
        pass

    diag: dict[str, int] = {
        "n_prompts": 0,
        "n_empty_ground_truth": 0,
        "n_skip_all_rollouts_empty": 0,
        "n_skip_no_mixed": 0,
        "n_mixed_pairs_ok": 0,
        "n_skip_empty_step_lp": 0,
        "n_jsonl_lines": 0,
        "n_jsonl_lines_correct": 0,
        "n_jsonl_lines_wrong": 0,
        "n_group_zero_viable_positions_correct": 0,
        "n_group_zero_viable_positions_wrong": 0,
    }

    rows = load_data(args.input_data, args.max_samples, args.seed)
    local_rows = [row for idx, row in enumerate(rows) if idx % world_size == rank]

    env_no_tqdm = os.environ.get("TQDM_DISABLE", "").strip().lower() in ("1", "true", "yes")
    use_tqdm = tqdm is not None and not args.no_progress and not env_no_tqdm and (
        args.progress_all_ranks or rank == 0
    )
    use_nested = bool(use_tqdm and args.progress_nested)
    bar_base = (rank * 4) if (use_tqdm and use_nested and args.progress_all_ranks) else (rank if use_tqdm else 0)
    shard_desc = f"shard{rank}"
    _tqdm_bar_fmt = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    )
    _tqdm_kw: dict[str, Any] = {
        "file": sys.stderr,
        "dynamic_ncols": True,
        "bar_format": _tqdm_bar_fmt,
    }

    prompt_iter = enumerate(local_rows)
    if use_tqdm:
        prompt_iter = tqdm(
            prompt_iter,
            total=len(local_rows),
            desc=f"{shard_desc} prompts",
            position=bar_base,
            leave=True,
            mininterval=1.0,
            **_tqdm_kw,
        )

    _run_t0 = time.perf_counter()
    _per_sample_cleared: set[int] = set()
    for local_i, row in prompt_iter:
        diag["n_prompts"] += 1
        global_idx = local_i * world_size + rank
        if per_sample_dir is not None and global_idx not in _per_sample_cleared:
            (per_sample_dir / f"sample_{global_idx:06d}.jsonl").unlink(missing_ok=True)
            _per_sample_cleared.add(global_idx)
        if rank == 0 and args.progress_echo:
            print(
                f"[pair_bias] rank0 >>> prompt {local_i + 1}/{len(local_rows)} "
                f"(global#{global_idx}) | cumulative {time.perf_counter() - _run_t0:.0f}s",
                file=sys.stderr,
                flush=True,
            )
        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        data_source = row.get("data_source", "math_dapo")
        ground_truth = _ground_truth_from_row(row)
        if not str(ground_truth).strip():
            diag["n_empty_ground_truth"] += 1

        rollouts: list[dict[str, Any]] = []
        rb_chunk = min(int(args.vllm_request_batch_chunk), int(args.rollouts_per_prompt))
        if use_nested and tqdm is not None:
            n_rb = (int(args.rollouts_per_prompt) + rb_chunk - 1) // rb_chunk
            _roll_chunks = tqdm(
                range(0, int(args.rollouts_per_prompt), rb_chunk),
                total=n_rb,
                desc=f"{shard_desc} rollout batches",
                position=bar_base + 1,
                leave=False,
                mininterval=0.2,
                **_tqdm_kw,
            )
        else:
            _roll_chunks = range(0, int(args.rollouts_per_prompt), rb_chunk)
        rollout_idx = 0
        for _ in _roll_chunks:
            bs = min(rb_chunk, int(args.rollouts_per_prompt) - rollout_idx)
            if bs <= 0:
                break
            batch_out = generate_rollouts_vllm_batched(
                llm=llm,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                logprobs_k=args.vllm_logprobs_topk,
                n_rollouts=bs,
                batch_chunk=rb_chunk,
            )
            for response_ids, scores in batch_out:
                if not response_ids:
                    rollout_idx += 1
                    continue
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                acc, _ = evaluate_solution_acc(
                    data_source=data_source, solution_str=response_text, ground_truth=ground_truth
                )
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
                rollout_idx += 1

        if not rollouts:
            diag["n_skip_all_rollouts_empty"] += 1
            continue

        correct = [r for r in rollouts if r["is_correct"]]
        wrong = [r for r in rollouts if not r["is_correct"]]
        if not correct or not wrong:
            diag["n_skip_no_mixed"] += 1
            continue

        diag["n_mixed_pairs_ok"] += 1

        # Deterministic selection: median-length candidate per group.
        def select_median_length(items: list[dict[str, Any]]) -> dict[str, Any]:
            items_sorted = sorted(items, key=lambda x: len(x["response_ids"]))
            return items_sorted[len(items_sorted) // 2]

        pair = [("correct", select_median_length(correct)), ("wrong", select_median_length(wrong))]
        for group_name, rr in pair:
            entropies = rr["entropies"]
            response_ids = rr["response_ids"]
            scores = rr["scores"]
            n_ent = len(entropies)
            n_via = sum(
                1
                for i in range(min(n_ent, len(scores)))
                if scores[i]
            )
            k_ratio = max(1, int(math.ceil(n_ent * float(args.top_entropy_ratio))))
            k_cap = int(args.max_positions_per_rollout)
            k_budget = min(k_cap, k_ratio) if k_cap > 0 else k_ratio
            positions = pick_top_entropy_positions(
                entropies, scores, float(args.top_entropy_ratio), int(args.max_positions_per_rollout)
            )
            n_pos_pick = len(positions)
            position_pick_meta = {
                "entropy_seq_len": int(n_ent),
                "viable_logprob_steps": int(n_via),
                "k_from_ratio_ceil": int(k_ratio),
                "k_budget_min_cap": int(k_budget),
                "k_effective_picked": int(n_pos_pick),
                "top_entropy_ratio": float(args.top_entropy_ratio),
                "max_positions_cap": int(k_cap),
            }
            if not positions:
                if group_name == "correct":
                    diag["n_group_zero_viable_positions_correct"] += 1
                else:
                    diag["n_group_zero_viable_positions_wrong"] += 1
            pos_iter = positions
            if use_nested and tqdm is not None:
                pos_iter = tqdm(
                    positions,
                    total=len(positions),
                    desc=f"{shard_desc} {group_name[:4]} pos",
                    position=bar_base + 2,
                    leave=False,
                    mininterval=0.1,
                    **_tqdm_kw,
                )
            for pos in pos_iter:
                if pos >= len(scores):
                    continue
                step_lp = scores[pos]
                if args.candidate_mode == "fixed":
                    tids, probs = topk_candidates_with_probs(step_lp, int(args.topk_alt))
                    cand_meta = {"k_topp": 0, "k_used": len(tids)}
                else:
                    tids, probs, cand_meta = topp_capped_candidates_with_probs(
                        step_lp,
                        float(args.candidate_top_p),
                        int(args.candidate_max_k),
                    )
                if not tids:
                    diag["n_skip_empty_step_lp"] += 1
                    continue

                mtok = max(1, args.max_new_tokens - pos - 1)
                prefix_selected = prompt_ids + response_ids[: pos + 1]
                prefixes_mc: list[list[int]] = [prefix_selected]
                for tid in tids:
                    prefixes_mc.append(prompt_ids + response_ids[:pos] + [int(tid)])
                bc = int(args.vllm_request_batch_chunk)
                n_req = len(prefixes_mc) * int(args.mc_m_samples)
                n_mc_chunks = (n_req + bc - 1) // bc
                sentence_stop_check = None
                if args.f_sentence_stop == "pysbd":
                    from sentence_stop_utils import make_pysbd_first_sentence_stop_check

                    sentence_stop_check = make_pysbd_first_sentence_stop_check()

                if args.f_continuation_mode == "first_sentence":
                    mc_prog = range(n_req)
                    if use_nested and tqdm is not None:
                        mc_prog = tqdm(
                            mc_prog,
                            total=n_req,
                            desc=f"{shard_desc} MC sentence",
                            position=bar_base + 3,
                            leave=False,
                            mininterval=0.2,
                            **_tqdm_kw,
                        )
                    Fs = estimate_F_mc_many_prefixes_vllm(
                        llm,
                        prefixes_mc,
                        int(args.mc_m_samples),
                        max_new_tokens=mtok,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        logprobs_k=args.vllm_logprobs_topk,
                        batch_chunk=bc,
                        chunk_starts_iter=None,
                        f_continuation_mode="first_sentence",
                        tokenizer=tokenizer,
                        sentence_chunk_max_tokens=int(args.f_sentence_chunk_max_tokens),
                        flat_progress_iter=mc_prog,
                        sentence_stop_check=sentence_stop_check,
                    )
                else:
                    chunk_starts = range(0, n_req, bc)
                    if use_nested and tqdm is not None:
                        chunk_starts = tqdm(
                            chunk_starts,
                            total=n_mc_chunks,
                            desc=f"{shard_desc} MC batched",
                            position=bar_base + 3,
                            leave=False,
                            mininterval=0.2,
                            **_tqdm_kw,
                        )
                    Fs = estimate_F_mc_many_prefixes_vllm(
                        llm,
                        prefixes_mc,
                        int(args.mc_m_samples),
                        max_new_tokens=mtok,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        logprobs_k=args.vllm_logprobs_topk,
                        batch_chunk=bc,
                        chunk_starts_iter=chunk_starts,
                    )
                f_selected = Fs[0]
                f_bar = sum(float(pr) * Fs[i + 1] for i, pr in enumerate(probs))
                f_alt_mcs = [float(Fs[i + 1]) for i in range(len(tids))]

                h_t = float(entropies[pos])
                bias_t = float(f_bar - f_selected)
                delta_hat = float(h_t + bias_t)

                ctx = build_token_context(
                    tokenizer, response_ids, int(pos), int(args.context_window_tokens)
                )
                alt_candidates: list[dict[str, Any]] = []
                for i, (tid, pr) in enumerate(zip(tids, probs, strict=False)):
                    tid_i = int(tid)
                    alt_candidates.append(
                        {
                            "token_id": tid_i,
                            "prob": float(pr),
                            "token_text": tokenizer.decode([tid_i], skip_special_tokens=True),
                            "f_mc": float(f_alt_mcs[i]),
                        }
                    )

                rec = {
                    "sample_index": global_idx,
                    "shard_rank": rank,
                    "group": group_name,
                    "rollout_index": rr["rollout_index"],
                    "token_index": int(pos),
                    "entropy_t": h_t,
                    "bar_F_t": float(f_bar),
                    "F_selected_mc": float(f_selected),
                    "f_alt_mc": f_alt_mcs,
                    "bias_t": bias_t,
                    "delta_hat_t": delta_hat,
                    "bias_over_entropy": float(bias_t / (h_t + 1e-8)),
                    "selected_token_id": int(response_ids[pos]),
                    "selected_token_text": tokenizer.decode(
                        [response_ids[pos]], skip_special_tokens=True
                    ),
                    "candidate_mode": str(args.candidate_mode),
                    "alt_k_topp": int(cand_meta["k_topp"]),
                    "alt_k_used": int(cand_meta["k_used"]),
                    "topk_alt_token_ids": [int(t) for t in tids],
                    "topk_alt_probs": [float(p) for p in probs],
                    "alt_candidates": alt_candidates,
                    "context": ctx,
                    "rollout_response_length": len(response_ids),
                    "num_picked_entropy_positions": int(n_pos_pick),
                    "mc_m_samples": int(args.mc_m_samples),
                    "f_continuation_mode": str(args.f_continuation_mode),
                    "f_sentence_chunk_max_tokens": int(args.f_sentence_chunk_max_tokens),
                    "f_sentence_stop": str(args.f_sentence_stop),
                    "vllm_request_batch_chunk": int(args.vllm_request_batch_chunk),
                    "data_source": data_source,
                    "ground_truth": ground_truth,
                    "prompt_text": prompt_text,
                    "position_pick_meta": position_pick_meta,
                }
                with open(part_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    diag["n_jsonl_lines"] += 1
                    if group_name == "correct":
                        diag["n_jsonl_lines_correct"] += 1
                    else:
                        diag["n_jsonl_lines_wrong"] += 1
                if per_sample_dir is not None:
                    spath = per_sample_dir / f"sample_{global_idx:06d}.jsonl"
                    with open(spath, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    diag_path = out_dir / f"pair_bias_diag_rank{rank}.json"
    diag["rank"] = rank
    diag["world_size"] = world_size
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

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
        if not args.no_per_sample_jsonl:
            _psd = out_dir / str(args.per_sample_jsonl_subdir).strip().replace("..", "_")
            print(
                f"[pair_bias] per-sample jsonl 目录: {_psd} (sample_XXXXXX.jsonl，每行一个 position×group)",
                file=sys.stderr,
                flush=True,
            )

        diag_merged: dict[str, int] = {}
        for r in range(world_size):
            dp = out_dir / f"pair_bias_diag_rank{r}.json"
            if not dp.exists():
                continue
            with open(dp, encoding="utf-8") as f:
                d = json.load(f)
            for k, v in d.items():
                if isinstance(v, bool) or k in ("rank", "world_size"):
                    continue
                if isinstance(v, (int, float)):
                    diag_merged[k] = diag_merged.get(k, 0) + int(v)
        if diag_merged:
            with open(out_dir / "pair_bias_diag_merged.json", "w", encoding="utf-8") as f:
                json.dump(diag_merged, f, ensure_ascii=False, indent=2)
            print(json.dumps({"pair_bias_diag_merged": diag_merged}, ensure_ascii=False, indent=2))

        if not merged:
            print(
                "[pair_bias] num_records=0: 请看 OUTPUT_DIR 下 pair_bias_diag_merged.json（各 rank 计数之和）。"
                " n_empty_ground_truth>0 表示 parquet 里没读到 GT；"
                " n_skip_no_mixed 高表示每条 prompt 上各 rollout 对错标签一致（需同时对错才会写 jsonl）；"
                " n_skip_all_rollouts_empty 表示 vLLM 未生成 token；"
                " n_skip_empty_step_lp 表示高熵步无 logprobs。",
                file=sys.stderr,
                flush=True,
            )
            npm = int(diag_merged.get("n_prompts", 0))
            nmix = int(diag_merged.get("n_skip_no_mixed", 0))
            if npm > 0 and nmix >= npm and int(args.max_new_tokens) < 2048:
                print(
                    f"[pair_bias] 提示: 当前 --max_new_tokens={args.max_new_tokens} 偏小。"
                    "R1/Distill 长思维链常在很后面才写 Answer:\\boxed{{}}，"
                    "过短会导致多条 rollout 同样截断、判分结果高度一致（看起来像「全对或全错」）。"
                    "GRPO 脚本里 data.max_response_length 常为 8192，建议分析时对齐，例如 "
                    "MAX_NEW_TOKENS=8192 或 --max_new_tokens 8192。",
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    main()

