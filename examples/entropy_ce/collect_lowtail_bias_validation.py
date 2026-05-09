#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
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
    _restore_torchrun_dist_env,
    _snapshot_and_clear_torchrun_dist_env,
    build_prompt_text,
    clamp_vllm_logprobs_topk,
    configure_cuda_visible_one_gpu_per_rank_for_vllm,
    entropy_from_logprobs_topk,
    estimate_F_mc_many_prefixes_vllm,
    evaluate_solution_acc,
    file_sync,
    init_dist,
    load_data,
    purge_all_torchrun_like_env_for_vllm_standalone,
    vllm_generate_quiet,
)
from infer_topk_f_mc_compare import _ground_truth_from_row, _topp_capped_from_step_logprobs

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _sample_rollout_with_logprobs(
    *,
    llm: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logprobs_k: int,
) -> tuple[list[int], list[dict[int, float]], list[float]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        logprobs=clamp_vllm_logprobs_topk(logprobs_k),
    )
    out = vllm_generate_quiet(llm, [TokensPrompt(prompt_token_ids=prompt_ids)], sp)
    o = out[0].outputs[0]
    response_ids = [int(t) for t in (o.token_ids or [])]
    step_lps: list[dict[int, float]] = []
    entropies: list[float] = []
    for step_lp in o.logprobs or []:
        d: dict[int, float] = {}
        for tid, info in step_lp.items():
            d[int(tid)] = float(info.logprob)
        step_lps.append(d)
        entropies.append(float(entropy_from_logprobs_topk(d)))
    while len(step_lps) < len(response_ids):
        step_lps.append({})
        entropies.append(float("nan"))
    return response_ids, step_lps[: len(response_ids)], entropies[: len(response_ids)]


def _select_high_entropy_positions(
    entropies: list[float],
    *,
    mode: str,
    threshold: float,
    top_ratio: float,
) -> list[int]:
    vals = np.asarray(entropies, dtype=np.float64)
    finite = np.where(np.isfinite(vals))[0]
    if finite.size == 0:
        return []
    if mode == "top_ratio":
        k = max(1, int(math.ceil(float(np.clip(top_ratio, 1e-9, 1.0)) * int(finite.size))))
        order = finite[np.argsort(vals[finite])[::-1]]
        return [int(x) for x in order[:k].tolist()]
    return [int(i) for i in finite.tolist() if float(vals[int(i)]) >= float(threshold)]


def _suffix_end_exclusive(
    response_ids: list[int],
    start: int,
    *,
    tokenizer: Any,
    suffix_mode: str,
    fixed_window_tokens: int,
    sentence_stop_check: Any | None,
) -> int:
    n = len(response_ids)
    if start >= n:
        return start
    if suffix_mode == "fixed_window":
        return min(n, start + max(1, int(fixed_window_tokens)))
    if suffix_mode == "sentence":
        from sentence_stop_utils import truncate_gen_ids_to_first_sentence

        k = int(truncate_gen_ids_to_first_sentence(response_ids[start:], tokenizer, sentence_stop_check))
        return min(n, start + max(0, k))
    return n


def _suffix_rate_from_realized(
    entropies: list[float],
    response_ids: list[int],
    t: int,
    *,
    tokenizer: Any,
    suffix_mode: str,
    fixed_window_tokens: int,
    sentence_stop_check: Any | None,
    min_suffix_tokens: int,
) -> tuple[float, int]:
    start = int(t) + 1
    end = _suffix_end_exclusive(
        response_ids,
        start,
        tokenizer=tokenizer,
        suffix_mode=suffix_mode,
        fixed_window_tokens=fixed_window_tokens,
        sentence_stop_check=sentence_stop_check,
    )
    suffix = [float(x) for x in entropies[start:end] if np.isfinite(float(x))]
    if len(suffix) < int(min_suffix_tokens):
        return float("nan"), int(len(suffix))
    return float(np.mean(np.asarray(suffix, dtype=np.float64))), int(len(suffix))


def _length_bucket_normalize(
    records: list[dict[str, Any]],
    *,
    bin_width: int,
    min_count: int,
    eps: float = 1e-6,
) -> None:
    buckets: dict[int, list[float]] = {}
    all_vals: list[float] = []
    bw = max(1, int(bin_width))
    for r in records:
        x = float(r.get("f_suffix_rate", float("nan")))
        l = int(r.get("suffix_tokens_used", 0) or 0)
        if not np.isfinite(x) or l <= 0:
            continue
        bid = (l - 1) // bw
        buckets.setdefault(bid, []).append(x)
        all_vals.append(x)
    global_mean = float(np.mean(all_vals)) if all_vals else 0.0
    global_std = float(np.std(all_vals)) if len(all_vals) > 1 else 0.0
    stats: dict[int, tuple[float, float, int]] = {}
    for bid, xs in buckets.items():
        arr = np.asarray(xs, dtype=np.float64)
        stats[bid] = (float(np.mean(arr)), float(np.std(arr)), int(arr.size))

    for r in records:
        x = float(r.get("f_suffix_rate", float("nan")))
        l = int(r.get("suffix_tokens_used", 0) or 0)
        if not np.isfinite(x) or l <= 0:
            r["f_suffix_rate_norm"] = float("nan")
            r["suffix_len_bucket"] = None
            r["suffix_len_bucket_count"] = 0
            r["suffix_len_bucket_fallback"] = True
            continue
        bid = (l - 1) // bw
        mean, std, cnt = stats.get(bid, (global_mean, global_std, 0))
        fallback = cnt < int(min_count)
        if fallback:
            mean, std, cnt = global_mean, global_std, len(all_vals)
        r["suffix_len_bucket"] = int(bid)
        r["suffix_len_bucket_count"] = int(cnt)
        r["suffix_len_bucket_fallback"] = bool(fallback)
        r["f_suffix_rate_norm"] = float((x - mean) / math.sqrt(std * std + eps))


def _assign_q(records: list[dict[str, Any]], key: str = "f_suffix_rate_norm") -> None:
    valid = [(i, float(r.get(key, float("nan")))) for i, r in enumerate(records)]
    valid = [(i, x) for i, x in valid if np.isfinite(x)]
    valid_sorted = sorted(valid, key=lambda z: z[1])
    n = len(valid_sorted)
    for rank, (i, _x) in enumerate(valid_sorted):
        records[i]["q"] = float(rank / (n - 1)) if n > 1 else 0.0
    for r in records:
        if "q" not in r:
            r["q"] = float("nan")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _compute_high_point_mc_in_batch(
    *,
    llm: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    response_ids: list[int],
    step_lps: list[dict[int, float]],
    entropies: list[float],
    high_positions: list[int],
    candidate_top_p: float,
    candidate_max_k: int,
    mc_m_samples: int,
    mc_temperature: float,
    mc_top_p: float,
    vllm_logprobs_topk: int,
    vllm_request_batch_chunk_mc: int,
    f_continuation_mode: str,
    f_sentence_max_new_tokens: int,
    sentence_stop_check: Any | None,
    suffix_mode: str,
    fixed_window_tokens: int,
    min_suffix_tokens: int,
) -> list[dict[str, Any]]:
    """Batch all high-point candidate prefixes for one prompt in one MC call."""
    jobs: list[dict[str, Any]] = []
    prefixes: list[list[int]] = []
    for t in high_positions:
        if t >= len(response_ids) or t >= len(step_lps):
            continue
        step_lp = step_lps[t]
        cands, cand_probs = _topp_capped_from_step_logprobs(
            step_lp,
            top_p=float(candidate_top_p),
            max_k=int(candidate_max_k),
        )
        chosen = int(response_ids[t])
        if chosen not in cands:
            cands = [chosen] + [int(x) for x in cands]
            cand_probs = [0.0] + [float(x) for x in cand_probs]
        suffix_rate, suffix_len = _suffix_rate_from_realized(
            entropies,
            response_ids,
            t,
            tokenizer=tokenizer,
            suffix_mode=str(suffix_mode),
            fixed_window_tokens=int(fixed_window_tokens),
            sentence_stop_check=sentence_stop_check,
            min_suffix_tokens=int(min_suffix_tokens),
        )
        job = {
            "t": int(t),
            "h_t": float(entropies[t]),
            "chosen": int(chosen),
            "cands": [int(x) for x in cands],
            "cand_probs": [float(x) for x in cand_probs],
            "suffix_rate": float(suffix_rate),
            "suffix_len": int(suffix_len),
        }
        jobs.append(job)
        prefixes.extend([prompt_ids + response_ids[:t] + [int(c)] for c in cands])
    if not jobs:
        return []

    f_all = estimate_F_mc_many_prefixes_vllm(
        llm=llm,
        prefixes=prefixes,
        m_samples=int(mc_m_samples),
        max_new_tokens=max(1, int(f_sentence_max_new_tokens)),
        temperature=float(mc_temperature),
        top_p=float(mc_top_p),
        logprobs_k=int(vllm_logprobs_topk),
        batch_chunk=int(vllm_request_batch_chunk_mc),
        f_continuation_mode=str(f_continuation_mode),
        tokenizer=tokenizer if str(f_continuation_mode) == "first_sentence" else None,
        f_sentence_max_new_tokens=int(f_sentence_max_new_tokens),
        sentence_stop_check=sentence_stop_check,
        normalize_by_continuation_length=True,
    )
    if len(f_all) != len(prefixes):
        return []

    out: list[dict[str, Any]] = []
    cur = 0
    for job in jobs:
        k = len(job["cands"])
        f_values = [float(x) for x in f_all[cur : cur + k]]
        cur += k
        if len(f_values) != k:
            continue
        chosen_i = int(job["cands"].index(job["chosen"]))
        f_bar = float(sum(float(p) * float(f_values[i]) for i, p in enumerate(job["cand_probs"])))
        f_real = float(f_values[chosen_i])
        bias = float(f_bar - f_real)
        out.append(
            {
                "response_t": int(job["t"]),
                "h_t": float(job["h_t"]),
                "chosen_token_id": int(job["chosen"]),
                "chosen_token_text": tokenizer.decode([int(job["chosen"])], skip_special_tokens=True),
                "f_bar": float(f_bar),
                "f_real": float(f_real),
                "bias": float(bias),
                "bias_pos": bool(bias > 0.0),
                "f_suffix_rate": float(job["suffix_rate"]),
                "suffix_tokens_used": int(job["suffix_len"]),
                "candidate_token_ids": [int(x) for x in job["cands"]],
                "candidate_probs": [float(x) for x in job["cand_probs"]],
                "candidate_f": [float(x) for x in f_values],
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect per-high-entropy-token FEPO bias diagnostics for low-tail validation."
    )
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--rollout_temperature", type=float, default=1.0)
    parser.add_argument("--rollout_top_p", type=float, default=0.95)
    parser.add_argument("--high_entropy_mode", choices=["threshold", "top_ratio"], default="threshold")
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    parser.add_argument("--high_entropy_top_ratio", type=float, default=0.1)
    parser.add_argument("--candidate_top_p", type=float, default=0.95)
    parser.add_argument("--candidate_max_k", type=int, default=20)
    parser.add_argument("--mc_m_samples", type=int, default=1)
    parser.add_argument("--mc_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=0.95)
    parser.add_argument("--f_continuation_mode", choices=["full", "first_sentence"], default="first_sentence")
    parser.add_argument("--f_sentence_max_new_tokens", type=int, default=256)
    parser.add_argument("--f_sentence_stop", choices=["simple", "pysbd"], default="simple")
    parser.add_argument("--suffix_mode", choices=["full", "sentence", "fixed_window"], default="sentence")
    parser.add_argument("--fixed_window_tokens", type=int, default=32)
    parser.add_argument("--min_suffix_tokens", type=int, default=5)
    parser.add_argument("--suffix_len_bin_width", type=int, default=3)
    parser.add_argument("--suffix_len_min_count", type=int, default=20)
    parser.add_argument("--vllm_logprobs_topk", type=int, default=20)
    parser.add_argument("--vllm_request_batch_chunk_mc", type=int, default=32)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--vllm_shard_rank", type=int, default=None)
    parser.add_argument("--vllm_shard_world_size", type=int, default=None)
    parser.add_argument(
        "--math_eval_backend",
        choices=["auto", "math_dapo", "math_verify"],
        default="math_verify",
        help="Correctness backend for math-like data. Default math_verify.",
    )
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    vllm_standalone = args.vllm_shard_rank is not None and args.vllm_shard_world_size is not None
    if (args.vllm_shard_rank is None) ^ (args.vllm_shard_world_size is None):
        raise SystemExit("Pass both --vllm_shard_rank and --vllm_shard_world_size, or neither.")
    if vllm_standalone:
        purge_all_torchrun_like_env_for_vllm_standalone()
    else:
        # torchrun 单机多卡时，让每个进程只看到一张 GPU，避免多个 vLLM 实例抢同一张卡。
        if "LOCAL_RANK" in os.environ:
            configure_cuda_visible_one_gpu_per_rank_for_vllm()

    _configure_vllm_multiprocessing_spawn()
    _configure_vllm_ipc_for_single_node()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, rank, world_size = init_dist(
        backend="vllm",
        rank_override=args.vllm_shard_rank if vllm_standalone else None,
        world_size_override=args.vllm_shard_world_size if vllm_standalone else None,
    )
    random.seed(int(args.seed) + rank)
    np.random.seed(int(args.seed) + rank)

    # Keep torchrun env stripped during the whole vLLM lifecycle so that any
    # lazily spawned vLLM subprocesses do not accidentally join torchrun's TCPStore.
    _snapshot_and_clear_torchrun_dist_env()
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
        "max_model_len": int(args.vllm_max_model_len),
        "enforce_eager": True,
    }
    if "seed" in inspect.signature(LLM).parameters:
        llm_kwargs["seed"] = int(args.seed) + rank
    llm = LLM(**llm_kwargs)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        for r in range(world_size):
            stale = out_dir / f"_done_lowtail_bias_rank{r}"
            if stale.exists():
                stale.unlink()

    rows = load_data(args.input_data, int(args.max_samples), int(args.seed))
    local_rows = [row for idx, row in enumerate(rows) if idx % world_size == rank]
    from sentence_stop_utils import completion_should_stop_after_first_sentence_simple

    sentence_stop_check = completion_should_stop_after_first_sentence_simple
    if args.f_sentence_stop == "pysbd":
        from sentence_stop_utils import make_pysbd_first_sentence_stop_check

        sentence_stop_check = make_pysbd_first_sentence_stop_check()

    point_records: list[dict[str, Any]] = []
    prompt_iter = enumerate(local_rows)
    if tqdm is not None and not args.no_progress:
        prompt_iter = tqdm(prompt_iter, total=len(local_rows), desc=f"rank{rank} prompts", dynamic_ncols=True)

    for local_i, row in prompt_iter:
        global_idx = local_i * world_size + rank
        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        response_ids, step_lps, entropies = _sample_rollout_with_logprobs(
            llm=llm,
            prompt_ids=prompt_ids,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.rollout_temperature),
            top_p=float(args.rollout_top_p),
            logprobs_k=max(int(args.vllm_logprobs_topk), int(args.candidate_max_k)),
        )
        high_positions = _select_high_entropy_positions(
            entropies,
            mode=str(args.high_entropy_mode),
            threshold=float(args.entropy_threshold),
            top_ratio=float(args.high_entropy_top_ratio),
        )
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        data_source = str(row.get("data_source", "math_dapo"))
        ground_truth = _ground_truth_from_row(row)
        response_correct, eval_info = evaluate_solution_acc(
            data_source=data_source,
            solution_str=response_text,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )
        point_stats = _compute_high_point_mc_in_batch(
            llm=llm,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            step_lps=step_lps,
            entropies=entropies,
            high_positions=high_positions,
            candidate_top_p=float(args.candidate_top_p),
            candidate_max_k=int(args.candidate_max_k),
            mc_m_samples=int(args.mc_m_samples),
            mc_temperature=float(args.mc_temperature),
            mc_top_p=float(args.mc_top_p),
            vllm_logprobs_topk=int(args.vllm_logprobs_topk),
            vllm_request_batch_chunk_mc=int(args.vllm_request_batch_chunk_mc),
            f_continuation_mode=str(args.f_continuation_mode),
            f_sentence_max_new_tokens=int(args.f_sentence_max_new_tokens),
            sentence_stop_check=sentence_stop_check,
            suffix_mode=str(args.suffix_mode),
            fixed_window_tokens=int(args.fixed_window_tokens),
            min_suffix_tokens=int(args.min_suffix_tokens),
        )
        for st in point_stats:
            point_records.append(
                {
                    "global_index": int(global_idx),
                    "local_index": int(local_i),
                    "data_source": data_source,
                    "ground_truth": ground_truth,
                    "response_correct": bool(response_correct),
                    "response_eval_info": eval_info,
                    "response_t": int(st["response_t"]),
                    "response_len": int(len(response_ids)),
                    "h_t": float(st["h_t"]),
                    "chosen_token_id": int(st["chosen_token_id"]),
                    "chosen_token_text": st["chosen_token_text"],
                    "f_bar": float(st["f_bar"]),
                    "f_real": float(st["f_real"]),
                    "bias": float(st["bias"]),
                    "bias_pos": bool(st["bias_pos"]),
                    "f_suffix_rate": float(st["f_suffix_rate"]),
                    "suffix_tokens_used": int(st["suffix_tokens_used"]),
                    "candidate_token_ids": [int(x) for x in st["candidate_token_ids"]],
                    "candidate_probs": [float(x) for x in st["candidate_probs"]],
                    "candidate_f": [float(x) for x in st["candidate_f"]],
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                }
            )

    _length_bucket_normalize(
        point_records,
        bin_width=int(args.suffix_len_bin_width),
        min_count=int(args.suffix_len_min_count),
    )
    _assign_q(point_records, key="f_suffix_rate_norm")

    part_path = out_dir / f"lowtail_bias_points_rank{rank}.jsonl"
    _write_jsonl(part_path, point_records)
    file_sync(out_dir, rank, world_size, tag="done_lowtail_bias")
    if rank == 0:
        merged: list[dict[str, Any]] = []
        for r in range(world_size):
            p = out_dir / f"lowtail_bias_points_rank{r}.jsonl"
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        merged.append(json.loads(line))
        _length_bucket_normalize(
            merged,
            bin_width=int(args.suffix_len_bin_width),
            min_count=int(args.suffix_len_min_count),
        )
        _assign_q(merged, key="f_suffix_rate_norm")
        _write_jsonl(out_dir / "lowtail_bias_points.jsonl", merged)
        summary = {
            "n_points": int(len(merged)),
            "bias_pos_rate": float(np.mean([1.0 if r.get("bias_pos") else 0.0 for r in merged])) if merged else None,
            "output": str(out_dir / "lowtail_bias_points.jsonl"),
            "q_definition": "rank percentile of length-bucket-normalized realized suffix entropy rate; low-tail has small q.",
            "bias_definition": "bias = f_bar - f_real; bias_pos means bias > 0.",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
