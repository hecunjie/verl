#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import time
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
    evaluate_solution_acc,
    file_sync,
    init_dist,
    load_data,
    purge_all_torchrun_like_env_for_vllm_standalone,
    vllm_generate_quiet,
)
from infer_topk_f_mc_compare import (
    _append_boxed_instruction,
    _build_bucket_group_estimator,
    _decode_one_policy,
    _ground_truth_from_row,
    _is_math_like_source,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else float("nan")


def _policy_from_mode(mode: str) -> str:
    if mode == "greedy":
        return "greedy_baseline"
    if mode == "sampling":
        return "sampling_baseline"
    if mode == "min_f_mc":
        return "min_f_mc"
    raise ValueError(f"unsupported mode: {mode}")


def _generate_many_full_vllm(
    *,
    llm: Any,
    prompt_ids: list[int],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_chunk: int,
) -> list[list[int]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    if n < 1:
        return []
    sp = SamplingParams(
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        logprobs=0,
    )
    out_ids: list[list[int]] = []
    for start in range(0, int(n), int(batch_chunk)):
        bs = min(int(batch_chunk), int(n) - start)
        prompts = [TokensPrompt(prompt_token_ids=prompt_ids) for _ in range(bs)]
        outputs = vllm_generate_quiet(llm, prompts, sp)
        if len(outputs) != bs:
            raise RuntimeError(f"vLLM batch size mismatch: expected {bs}, got {len(outputs)}")
        for out_req in outputs:
            out_ids.append([int(x) for x in list(out_req.outputs[0].token_ids)])
    return out_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-mode multi-sample decoding and pass@k evaluation.")
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", choices=["greedy", "sampling", "min_f_mc"], required=True)
    parser.add_argument("--num_samples_per_prompt", type=int, default=32)
    parser.add_argument("--pass_k_small", type=int, default=4)
    parser.add_argument("--pass_k_large", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm_seed", type=int, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    parser.add_argument("--candidate_top_p", type=float, default=0.95)
    parser.add_argument("--candidate_max_k", type=int, default=5)
    parser.add_argument(
        "--selection_f_mode",
        choices=["mc", "greedy_path", "lookahead_1step", "bucket_group_estimate"],
        default="greedy_path",
    )
    parser.add_argument("--max_branch_steps", type=int, default=0)

    parser.add_argument("--mc_m_samples", type=int, default=1)
    parser.add_argument("--mc_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=0.95)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_top_p", type=float, default=0.95)
    parser.add_argument("--minf_nonbranch_mode", choices=["greedy", "sampling"], default="greedy")
    parser.add_argument("--bias_metrics_mode", choices=["raw", "length_normalized"], default="length_normalized")
    parser.add_argument("--f_continuation_mode", choices=["full", "first_sentence"], default="first_sentence")
    parser.add_argument("--f_sentence_max_new_tokens", type=int, default=256)
    parser.add_argument("--f_sentence_stop", choices=["simple", "pysbd"], default="simple")

    parser.add_argument("--vllm_logprobs_topk", type=int, default=20)
    parser.add_argument("--vllm_request_batch_chunk", type=int, default=64)
    parser.add_argument("--vllm_request_batch_chunk_mc", type=int, default=0)
    parser.add_argument("--bucket_group_rollouts", type=int, default=16)
    parser.add_argument("--bucket_num_bins", type=int, default=100)
    parser.add_argument("--bucket_min_points_per_bin", type=int, default=4)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--progress_all_ranks", action="store_true")
    parser.add_argument("--progress_echo", action="store_true")

    parser.add_argument("--vllm_shard_rank", type=int, default=None)
    parser.add_argument("--vllm_shard_world_size", type=int, default=None)
    parser.add_argument("--math_eval_backend", choices=["auto", "math_dapo", "math_verify"], default="auto")
    parser.add_argument("--force_boxed_answer_instruction", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.num_samples_per_prompt < 1:
        raise SystemExit("--num_samples_per_prompt must be >=1.")
    if args.pass_k_small < 1 or args.pass_k_large < 1:
        raise SystemExit("--pass_k_small/pass_k_large must be >=1.")
    if int(args.vllm_request_batch_chunk_mc) != 0 and int(args.vllm_request_batch_chunk_mc) < 1:
        raise SystemExit("--vllm_request_batch_chunk_mc must be 0 or >=1.")

    vllm_standalone = args.vllm_shard_rank is not None and args.vllm_shard_world_size is not None
    if (args.vllm_shard_rank is None) ^ (args.vllm_shard_world_size is None):
        raise SystemExit("Pass both --vllm_shard_rank and --vllm_shard_world_size, or neither.")
    if vllm_standalone:
        purge_all_torchrun_like_env_for_vllm_standalone()

    mc_batch_chunk = (
        int(args.vllm_request_batch_chunk_mc)
        if int(args.vllm_request_batch_chunk_mc) > 0
        else int(args.vllm_request_batch_chunk)
    )

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
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    vllm_seed_use = int(args.vllm_seed) if args.vllm_seed is not None else int(args.seed + rank)

    dist_snap = _snapshot_and_clear_torchrun_dist_env()
    try:
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
            llm_kwargs["seed"] = vllm_seed_use
        llm = LLM(**llm_kwargs)
    finally:
        _restore_torchrun_dist_env(dist_snap)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        for r in range(world_size):
            stale = out_dir / f"_done_passk_mode_rank{r}"
            if stale.exists():
                stale.unlink()
    part_path = out_dir / f"passk_mode_rank{rank}.jsonl"
    part_path.write_text("", encoding="utf-8")

    rows = load_data(args.input_data, args.max_samples, args.seed)
    local_rows = [row for idx, row in enumerate(rows) if idx % world_size == rank]
    policy = _policy_from_mode(str(args.mode))

    pass_small_all: list[float] = []
    pass_large_all: list[float] = []
    mean_large_all: list[float] = []

    use_tqdm = tqdm is not None and not args.no_progress and (args.progress_all_ranks or rank == 0)
    prompt_iter: Any = enumerate(local_rows)
    if use_tqdm:
        prompt_iter = tqdm(prompt_iter, total=len(local_rows), desc=f"mode={args.mode} rank{rank}", dynamic_ncols=True)

    run_t0 = 0.0
    if args.progress_echo and rank == 0:
        run_t0 = time.perf_counter()

    for local_i, row in prompt_iter:
        global_idx = local_i * world_size + rank
        if args.progress_echo and rank == 0:
            elapsed = time.perf_counter() - run_t0
            print(
                f"[passk_mode] rank0 prompt {local_i + 1}/{len(local_rows)} global#{global_idx} elapsed={elapsed:.1f}s",
                flush=True,
            )

        data_source = row.get("data_source", "math_dapo")
        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        if args.force_boxed_answer_instruction and _is_math_like_source(str(data_source)):
            prompt_text = _append_boxed_instruction(prompt_text)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        ground_truth = _ground_truth_from_row(row)

        bucket_estimator: dict[str, Any] | None = None
        if policy == "min_f_mc" and str(args.selection_f_mode) == "bucket_group_estimate":
            bucket_estimator = _build_bucket_group_estimator(
                llm=llm,
                prompt_ids=prompt_ids,
                n_rollouts=int(args.bucket_group_rollouts),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.mc_temperature),
                top_p=float(args.mc_top_p),
                logprobs_k=int(args.vllm_logprobs_topk),
                batch_chunk=int(args.vllm_request_batch_chunk),
                num_bins=int(args.bucket_num_bins),
                min_points_per_bin=int(args.bucket_min_points_per_bin),
            )

        sample_correct: list[float] = []
        sample_lens: list[int] = []
        if policy in {"greedy_baseline", "sampling_baseline"}:
            # Fast path: fully batched decode for pass@k workloads.
            if policy == "greedy_baseline":
                temp_use, top_p_use = 0.0, 1.0
            else:
                temp_use, top_p_use = float(args.sampling_temperature), float(args.sampling_top_p)
            batched_ids = _generate_many_full_vllm(
                llm=llm,
                prompt_ids=prompt_ids,
                n=int(args.num_samples_per_prompt),
                max_new_tokens=int(args.max_new_tokens),
                temperature=temp_use,
                top_p=top_p_use,
                batch_chunk=int(args.vllm_request_batch_chunk),
            )
            for response_ids in batched_ids:
                text = tokenizer.decode(response_ids, skip_special_tokens=True)
                ok, _eval = evaluate_solution_acc(
                    data_source=data_source,
                    solution_str=text,
                    ground_truth=ground_truth,
                    math_eval_backend=str(args.math_eval_backend),
                )
                sample_correct.append(1.0 if ok else 0.0)
                sample_lens.append(len(response_ids))
        else:
            # min_f_mc currently keeps per-sample decoding logic.
            for sample_idx in range(int(args.num_samples_per_prompt)):
                response_ids, _trace = _decode_one_policy(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    policy=policy,
                    entropy_threshold=float(args.entropy_threshold),
                    candidate_top_p=float(args.candidate_top_p),
                    candidate_max_k=int(args.candidate_max_k),
                    selection_f_mode=str(args.selection_f_mode),
                    max_new_tokens=int(args.max_new_tokens),
                    mc_m_samples=int(args.mc_m_samples),
                    mc_temperature=float(args.mc_temperature),
                    mc_top_p=float(args.mc_top_p),
                    sampling_temperature=float(args.sampling_temperature),
                    sampling_top_p=float(args.sampling_top_p),
                    minf_nonbranch_mode=str(args.minf_nonbranch_mode),
                    vllm_logprobs_topk=int(args.vllm_logprobs_topk),
                    vllm_request_batch_chunk=int(args.vllm_request_batch_chunk),
                    vllm_request_batch_chunk_mc=int(mc_batch_chunk),
                    f_continuation_mode=str(args.f_continuation_mode),
                    f_sentence_max_new_tokens=int(args.f_sentence_max_new_tokens),
                    f_sentence_stop=str(args.f_sentence_stop),
                    normalize_by_continuation_length=(args.bias_metrics_mode == "length_normalized"),
                    max_branch_steps=int(args.max_branch_steps),
                    rng=random.Random(args.seed + rank + global_idx * 100000 + sample_idx),
                    eos_token_id=tokenizer.eos_token_id,
                    show_nested_progress=False,
                    bucket_group_estimator=bucket_estimator,
                )
                text = tokenizer.decode(response_ids, skip_special_tokens=True)
                ok, _eval = evaluate_solution_acc(
                    data_source=data_source,
                    solution_str=text,
                    ground_truth=ground_truth,
                    math_eval_backend=str(args.math_eval_backend),
                )
                sample_correct.append(1.0 if ok else 0.0)
                sample_lens.append(len(response_ids))

        k_small = min(int(args.pass_k_small), len(sample_correct))
        k_large = min(int(args.pass_k_large), len(sample_correct))
        pass_small = 1.0 if any(x > 0.5 for x in sample_correct[:k_small]) else 0.0
        pass_large = 1.0 if any(x > 0.5 for x in sample_correct[:k_large]) else 0.0
        mean_large = _mean(sample_correct[:k_large]) if k_large > 0 else float("nan")

        pass_small_all.append(pass_small)
        pass_large_all.append(pass_large)
        mean_large_all.append(mean_large)

        rec = {
            "sample_index": int(global_idx),
            "shard_rank": int(rank),
            "mode": str(args.mode),
            "policy": policy,
            "num_samples_per_prompt": int(args.num_samples_per_prompt),
            "pass_k_small": int(args.pass_k_small),
            "pass_k_large": int(args.pass_k_large),
            "pass_at_small": float(pass_small),
            "pass_at_large": float(pass_large),
            "mean_at_large": float(mean_large),
            "sample_correct": [float(x) for x in sample_correct],
            "avg_response_len": _mean([float(x) for x in sample_lens]),
        }
        with open(part_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    file_sync(out_dir=out_dir, rank=rank, world_size=world_size, tag="done_passk_mode")

    if rank == 0:
        merged: list[dict[str, Any]] = []
        for r in range(world_size):
            p = out_dir / f"passk_mode_rank{r}.jsonl"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged.append(json.loads(line))

        with open(out_dir / "passk_mode_merged.jsonl", "w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        pass_small_vals = [float(x["pass_at_small"]) for x in merged]
        pass_large_vals = [float(x["pass_at_large"]) for x in merged]
        mean_large_vals = [float(x["mean_at_large"]) for x in merged]

        summary = {
            "num_prompts": int(len(merged)),
            "mode": str(args.mode),
            "policy": policy,
            "num_samples_per_prompt": int(args.num_samples_per_prompt),
            f"pass@{int(args.pass_k_small)}": _mean(pass_small_vals),
            f"pass@{int(args.pass_k_large)}": _mean(pass_large_vals),
            f"mean@{int(args.pass_k_large)}": _mean(mean_large_vals),
            "config": {
                "seed": int(args.seed),
                "vllm_seed": int(vllm_seed_use),
                "math_eval_backend": str(args.math_eval_backend),
                "sampling_temperature": float(args.sampling_temperature),
                "sampling_top_p": float(args.sampling_top_p),
                "minf_nonbranch_mode": str(args.minf_nonbranch_mode),
                "selection_f_mode": str(args.selection_f_mode),
                "mc_m_samples": int(args.mc_m_samples),
                "vllm_request_batch_chunk": int(args.vllm_request_batch_chunk),
                "vllm_request_batch_chunk_mc": int(mc_batch_chunk),
            },
        }
        with open(out_dir / "passk_mode_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

