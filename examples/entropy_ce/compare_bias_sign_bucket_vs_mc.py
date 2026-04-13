#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
    entropy_from_logprobs_topk,
    estimate_F_mc_many_prefixes_vllm,
    file_sync,
    init_dist,
    load_data,
    purge_all_torchrun_like_env_for_vllm_standalone,
)
from infer_topk_f_mc_compare import (
    _build_bucket_group_estimator,
    _bucket_estimate_bar_f,
    _candidate_lookahead_1step_entropy,
    _ground_truth_from_row,
    _step_logprobs_vllm,
    _topp_capped_from_step_logprobs,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _sign(x: float, eps: float = 1e-8) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _mean_ratio(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sign agreement of bias term: bucket_group_estimate vs MC=128."
    )
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    parser.add_argument("--candidate_top_p", type=float, default=0.95)
    parser.add_argument("--candidate_max_k", type=int, default=5)
    parser.add_argument("--max_branch_steps", type=int, default=64, help="<=0 means no cap.")

    parser.add_argument("--mc_m_samples_ref", type=int, default=128, help="MC samples for reference sign.")
    parser.add_argument("--mc_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=0.95)
    parser.add_argument("--bias_metrics_mode", choices=["raw", "length_normalized"], default="length_normalized")
    parser.add_argument("--f_continuation_mode", choices=["full", "first_sentence"], default="first_sentence")
    parser.add_argument("--f_sentence_max_new_tokens", type=int, default=256)
    parser.add_argument("--f_sentence_stop", choices=["simple", "pysbd"], default="simple")

    parser.add_argument("--vllm_logprobs_topk", type=int, default=20)
    parser.add_argument("--vllm_request_batch_chunk", type=int, default=64)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)

    parser.add_argument("--bucket_group_rollouts", type=int, default=16)
    parser.add_argument("--bucket_num_bins", type=int, default=100)
    parser.add_argument("--bucket_min_points_per_bin", type=int, default=4)

    parser.add_argument("--save_traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--progress_all_ranks", action="store_true")
    parser.add_argument("--progress_echo", action="store_true")

    parser.add_argument("--vllm_shard_rank", type=int, default=None)
    parser.add_argument("--vllm_shard_world_size", type=int, default=None)
    args = parser.parse_args()

    if int(args.candidate_max_k) < 2:
        raise SystemExit("--candidate_max_k must be >=2.")
    if int(args.max_new_tokens) < 1:
        raise SystemExit("--max_new_tokens must be >=1.")
    if int(args.vllm_request_batch_chunk) < 1:
        raise SystemExit("--vllm_request_batch_chunk must be >=1.")
    if int(args.mc_m_samples_ref) < 1:
        raise SystemExit("--mc_m_samples_ref must be >=1.")
    if int(args.bucket_group_rollouts) < 1:
        raise SystemExit("--bucket_group_rollouts must be >=1.")
    if int(args.bucket_num_bins) < 2:
        raise SystemExit("--bucket_num_bins must be >=2.")
    if int(args.bucket_min_points_per_bin) < 1:
        raise SystemExit("--bucket_min_points_per_bin must be >=1.")

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

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"sign_compare_rank{rank}.jsonl"
    part_path.write_text("", encoding="utf-8")

    rows = load_data(args.input_data, args.max_samples, args.seed)
    local_rows = [row for idx, row in enumerate(rows) if idx % world_size == rank]

    env_no_tqdm = os.environ.get("TQDM_DISABLE", "").strip().lower() in ("1", "true", "yes")
    use_tqdm = tqdm is not None and not args.no_progress and not env_no_tqdm and (
        args.progress_all_ranks or rank == 0
    )

    p_iter = enumerate(local_rows)
    if use_tqdm:
        p_iter = tqdm(
            p_iter,
            total=len(local_rows),
            desc=f"shard{rank} prompts",
            dynamic_ncols=True,
            leave=True,
            position=rank if args.progress_all_ranks else 0,
        )

    summary_cnt: dict[str, int] = {
        "n_prompts": 0,
        "n_eval_steps": 0,
        "n_eval_candidates": 0,
        "n_mc_nonzero_candidates": 0,
        "n_match_all_candidates": 0,
        "n_match_mc_nonzero_candidates": 0,
        "n_eval_steps_chosen": 0,
        "n_mc_nonzero_steps_chosen": 0,
        "n_match_steps_chosen": 0,
        "n_match_steps_chosen_mc_nonzero": 0,
    }

    sentence_stop_check = None
    if str(args.f_sentence_stop) == "pysbd":
        from sentence_stop_utils import make_pysbd_first_sentence_stop_check

        sentence_stop_check = make_pysbd_first_sentence_stop_check()

    for local_i, row in p_iter:
        global_idx = local_i * world_size + rank
        summary_cnt["n_prompts"] += 1
        if args.progress_echo and rank == 0:
            print(f"[sign_compare] prompt {local_i + 1}/{len(local_rows)} global#{global_idx}", flush=True)

        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
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

        response_ids: list[int] = []
        entropy_hist: list[float] = []
        step_records: list[dict[str, Any]] = []
        branch_count = 0

        for step_idx in range(int(args.max_new_tokens)):
            prefix = prompt_ids + response_ids
            greedy_token, step_lp = _step_logprobs_vllm(
                llm=llm,
                prefix_ids=prefix,
                logprobs_k=max(int(args.vllm_logprobs_topk), int(args.candidate_max_k)),
            )
            if greedy_token is None:
                break

            entropy_t = float(entropy_from_logprobs_topk(step_lp))
            cands, cand_probs = _topp_capped_from_step_logprobs(
                step_lp,
                top_p=float(args.candidate_top_p),
                max_k=int(args.candidate_max_k),
            )
            if not cands:
                cands = [int(greedy_token)]
                cand_probs = [1.0]

            should_eval = (
                entropy_t >= float(args.entropy_threshold)
                and len(cands) >= 2
                and (int(args.max_branch_steps) <= 0 or branch_count < int(args.max_branch_steps))
            )
            if should_eval:
                remaining = int(args.max_new_tokens) - step_idx - 1
                if remaining > 0:
                    prefixes = [prompt_ids + response_ids + [int(t)] for t in cands]
                    f_mc = estimate_F_mc_many_prefixes_vllm(
                        llm=llm,
                        prefixes=prefixes,
                        m_samples=int(args.mc_m_samples_ref),
                        max_new_tokens=remaining,
                        temperature=float(args.mc_temperature),
                        top_p=float(args.mc_top_p),
                        logprobs_k=int(args.vllm_logprobs_topk),
                        batch_chunk=int(args.vllm_request_batch_chunk),
                        f_continuation_mode=str(args.f_continuation_mode),
                        tokenizer=tokenizer if str(args.f_continuation_mode) == "first_sentence" else None,
                        f_sentence_max_new_tokens=int(args.f_sentence_max_new_tokens),
                        sentence_stop_check=sentence_stop_check,
                        normalize_by_continuation_length=(str(args.bias_metrics_mode) == "length_normalized"),
                    )
                    if len(f_mc) == len(cands):
                        fbar_mc = float(sum(float(p) * float(v) for p, v in zip(cand_probs, f_mc, strict=False)))
                        sign_mc = [_sign(fbar_mc - float(v)) for v in f_mc]

                        prefix_sum = float(sum(entropy_hist) + entropy_t)
                        prefix_len = len(entropy_hist) + 1
                        f_bucket: list[float] = []
                        for t in cands:
                            h1 = _candidate_lookahead_1step_entropy(
                                llm=llm,
                                prefix_ids=(prompt_ids + response_ids),
                                candidate_token_id=int(t),
                                logprobs_k=int(args.vllm_logprobs_topk),
                            )
                            p_rate = (prefix_sum + float(h1)) / float(prefix_len + 1)
                            f_bucket.append(float(_bucket_estimate_bar_f(bucket_estimator, p_rate)))
                        fbar_bucket = float(
                            sum(float(p) * float(v) for p, v in zip(cand_probs, f_bucket, strict=False))
                        )
                        sign_bucket = [_sign(fbar_bucket - float(v)) for v in f_bucket]

                        chosen_idx = 0
                        if int(greedy_token) in cands:
                            chosen_idx = cands.index(int(greedy_token))

                        cand_match = [int(a == b) for a, b in zip(sign_bucket, sign_mc, strict=False)]
                        nz_mask = [int(s != 0) for s in sign_mc]
                        nz_match = [m for m, z in zip(cand_match, nz_mask, strict=False) if z == 1]

                        summary_cnt["n_eval_steps"] += 1
                        summary_cnt["n_eval_candidates"] += int(len(cands))
                        summary_cnt["n_match_all_candidates"] += int(sum(cand_match))
                        summary_cnt["n_mc_nonzero_candidates"] += int(sum(nz_mask))
                        summary_cnt["n_match_mc_nonzero_candidates"] += int(sum(nz_match))
                        summary_cnt["n_eval_steps_chosen"] += 1
                        summary_cnt["n_match_steps_chosen"] += int(sign_bucket[chosen_idx] == sign_mc[chosen_idx])
                        if sign_mc[chosen_idx] != 0:
                            summary_cnt["n_mc_nonzero_steps_chosen"] += 1
                            summary_cnt["n_match_steps_chosen_mc_nonzero"] += int(
                                sign_bucket[chosen_idx] == sign_mc[chosen_idx]
                            )

                        if bool(args.save_traces):
                            step_records.append(
                                {
                                    "step_index": int(step_idx),
                                    "entropy_t": float(entropy_t),
                                    "chosen_token": int(greedy_token),
                                    "chosen_text": tokenizer.decode([int(greedy_token)], skip_special_tokens=True),
                                    "candidates": [int(x) for x in cands],
                                    "candidate_probs_renorm_topp": [float(x) for x in cand_probs],
                                    "f_mc_128": [float(x) for x in f_mc],
                                    "f_bar_mc_128": float(fbar_mc),
                                    "bias_sign_mc_128": [int(x) for x in sign_mc],
                                    "f_bucket_proxy": [float(x) for x in f_bucket],
                                    "f_bar_bucket_proxy": float(fbar_bucket),
                                    "bias_sign_bucket_proxy": [int(x) for x in sign_bucket],
                                    "chosen_idx": int(chosen_idx),
                                    "chosen_sign_mc_128": int(sign_mc[chosen_idx]),
                                    "chosen_sign_bucket_proxy": int(sign_bucket[chosen_idx]),
                                    "chosen_sign_match": bool(sign_mc[chosen_idx] == sign_bucket[chosen_idx]),
                                }
                            )
                        branch_count += 1

            response_ids.append(int(greedy_token))
            entropy_hist.append(float(entropy_t))
            if tokenizer.eos_token_id is not None and int(greedy_token) == int(tokenizer.eos_token_id):
                break

        rec: dict[str, Any] = {
            "sample_index": int(global_idx),
            "shard_rank": int(rank),
            "ground_truth": _ground_truth_from_row(row),
            "entropy_threshold": float(args.entropy_threshold),
            "candidate_top_p": float(args.candidate_top_p),
            "candidate_max_k": int(args.candidate_max_k),
            "max_new_tokens": int(args.max_new_tokens),
            "max_branch_steps": int(args.max_branch_steps),
            "mc_m_samples_ref": int(args.mc_m_samples_ref),
            "bucket_group_rollouts": int(args.bucket_group_rollouts),
            "bucket_num_bins": int(args.bucket_num_bins),
            "bucket_min_points_per_bin": int(args.bucket_min_points_per_bin),
            "num_eval_steps": int(len(step_records)) if bool(args.save_traces) else int(branch_count),
        }
        if bool(args.save_traces):
            rec["trace_sign_compare"] = step_records
        _append_jsonl(part_path, rec)

    with open(out_dir / f"sign_compare_counter_rank{rank}.json", "w", encoding="utf-8") as f:
        json.dump(summary_cnt, f, ensure_ascii=False, indent=2)

    file_sync(out_dir=out_dir, rank=rank, world_size=world_size, tag="done_sign_compare")

    if rank == 0:
        merged: list[dict[str, Any]] = []
        for r in range(world_size):
            p = out_dir / f"sign_compare_rank{r}.jsonl"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged.append(json.loads(line))
        with open(out_dir / "sign_compare_merged.jsonl", "w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total = dict(summary_cnt)
        for r in range(1, world_size):
            # each rank prints a compact counter json; merge if present
            cp = out_dir / f"sign_compare_counter_rank{r}.json"
            if not cp.exists():
                continue
            with open(cp, encoding="utf-8") as f:
                other = json.load(f)
            for k in total:
                total[k] += int(other.get(k, 0))

        summary = {
            "num_prompts": int(total["n_prompts"]),
            "num_eval_steps": int(total["n_eval_steps"]),
            "num_eval_candidates": int(total["n_eval_candidates"]),
            "candidate_sign_match_acc_all": _mean_ratio(
                int(total["n_match_all_candidates"]), int(total["n_eval_candidates"])
            ),
            "candidate_sign_match_acc_mc_nonzero_only": _mean_ratio(
                int(total["n_match_mc_nonzero_candidates"]), int(total["n_mc_nonzero_candidates"])
            ),
            "chosen_sign_match_acc_all": _mean_ratio(
                int(total["n_match_steps_chosen"]), int(total["n_eval_steps_chosen"])
            ),
            "chosen_sign_match_acc_mc_nonzero_only": _mean_ratio(
                int(total["n_match_steps_chosen_mc_nonzero"]), int(total["n_mc_nonzero_steps_chosen"])
            ),
            "counts": total,
            "config": {
                "entropy_threshold": float(args.entropy_threshold),
                "candidate_top_p": float(args.candidate_top_p),
                "candidate_max_k": int(args.candidate_max_k),
                "max_new_tokens": int(args.max_new_tokens),
                "max_branch_steps": int(args.max_branch_steps),
                "mc_m_samples_ref": int(args.mc_m_samples_ref),
                "mc_temperature": float(args.mc_temperature),
                "mc_top_p": float(args.mc_top_p),
                "f_continuation_mode": str(args.f_continuation_mode),
                "f_sentence_max_new_tokens": int(args.f_sentence_max_new_tokens),
                "f_sentence_stop": str(args.f_sentence_stop),
                "bias_metrics_mode": str(args.bias_metrics_mode),
                "bucket_group_rollouts": int(args.bucket_group_rollouts),
                "bucket_num_bins": int(args.bucket_num_bins),
                "bucket_min_points_per_bin": int(args.bucket_min_points_per_bin),
            },
        }
        with open(out_dir / "sign_compare_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()

