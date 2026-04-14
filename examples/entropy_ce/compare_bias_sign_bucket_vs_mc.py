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
    _ground_truth_from_row,
    _sample_group_rollouts_for_bucket,
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


def _build_bucket_estimator_with_key_mode(
    *,
    hs_rollouts: list[list[float]],
    num_bins: int,
    min_points_per_bin: int,
    prefix_key_mode: str,
) -> dict[str, Any]:
    b = max(2, int(num_bins))
    prefix_keys: list[float] = []
    suffix_rates: list[float] = []
    for hs in hs_rollouts:
        n = len(hs)
        if n < 2:
            continue
        suffix_sum = np.cumsum(np.array(hs[::-1], dtype=np.float64))[::-1]
        pref_sum = 0.0
        for u in range(1, n):
            pref_sum += float(hs[u - 1])
            if prefix_key_mode == "sum":
                key = pref_sum
            else:
                key = pref_sum / float(u)
            suffix_rate = float(suffix_sum[u] / float(n - u))
            prefix_keys.append(float(key))
            suffix_rates.append(float(suffix_rate))

    if not prefix_keys:
        return {
            "degenerate": True,
            "edges": [0.0 for _ in range(b + 1)],
            "means": [0.0 for _ in range(b)],
            "valid": [True for _ in range(b)],
            "global_mean_suffix_rate": 0.0,
        }

    p = np.array(prefix_keys, dtype=np.float64)
    f = np.array(suffix_rates, dtype=np.float64)
    pmin = float(np.min(p))
    pmax = float(np.max(p))
    global_mean = float(np.mean(f))
    if (not np.isfinite(pmin)) or (not np.isfinite(pmax)) or abs(pmax - pmin) < 1e-12:
        return {
            "degenerate": True,
            "edges": [0.0 for _ in range(b + 1)],
            "means": [float(global_mean) for _ in range(b)],
            "valid": [True for _ in range(b)],
            "global_mean_suffix_rate": float(global_mean),
        }

    edges = np.linspace(pmin, pmax, b + 1, dtype=np.float64)
    sums = np.zeros(b, dtype=np.float64)
    cnts = np.zeros(b, dtype=np.int64)
    for pi, fi in zip(p.tolist(), f.tolist()):
        idx = int(np.searchsorted(edges, pi, side="right") - 1)
        idx = min(max(idx, 0), b - 1)
        sums[idx] += float(fi)
        cnts[idx] += 1

    means = np.zeros(b, dtype=np.float64)
    valid = np.zeros(b, dtype=np.bool_)
    min_pts = max(1, int(min_points_per_bin))
    for i in range(b):
        if int(cnts[i]) >= min_pts:
            means[i] = float(sums[i] / float(cnts[i]))
            valid[i] = True

    return {
        "degenerate": False,
        "edges": [float(x) for x in edges.tolist()],
        "means": [float(x) for x in means.tolist()],
        "valid": [bool(x) for x in valid.tolist()],
        "global_mean_suffix_rate": float(global_mean),
    }


def _bucket_lookup(estimator: dict[str, Any], prefix_key: float) -> float:
    gm = float(estimator.get("global_mean_suffix_rate", 0.0))
    if bool(estimator.get("degenerate", False)):
        return gm
    edges = estimator.get("edges")
    means = estimator.get("means")
    valid = estimator.get("valid")
    if not isinstance(edges, list) or not isinstance(means, list) or not isinstance(valid, list):
        return gm
    b = len(means)
    if b <= 0:
        return gm
    idx = int(np.searchsorted(np.array(edges, dtype=np.float64), float(prefix_key), side="right") - 1)
    idx = min(max(idx, 0), b - 1)
    if bool(valid[idx]):
        return float(means[idx])
    nearest_i = None
    nearest_d = None
    for i, ok in enumerate(valid):
        if not ok:
            continue
        d = abs(i - idx)
        if nearest_d is None or d < nearest_d:
            nearest_d = d
            nearest_i = i
    if nearest_i is None:
        return gm
    return float(means[int(nearest_i)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare real-token bias-sign: bucket estimate vs MC reference."
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
    parser.add_argument(
        "--bucket_prefix_key_mode",
        choices=["sum", "rate"],
        default="sum",
        help="Prefix key for bucket retrieval: sum (recommended) or rate.",
    )

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
        "n_mc_nonzero_steps": 0,
        "n_match_steps": 0,
        "n_match_steps_if_flip": 0,
        "n_skipped_chosen_not_in_candidates": 0,
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
        hs_rollouts = _sample_group_rollouts_for_bucket(
            llm=llm,
            prompt_ids=prompt_ids,
            n_rollouts=int(args.bucket_group_rollouts),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.mc_temperature),
            top_p=float(args.mc_top_p),
            logprobs_k=int(args.vllm_logprobs_topk),
            batch_chunk=int(args.vllm_request_batch_chunk),
        )
        bucket_estimator = _build_bucket_estimator_with_key_mode(
            hs_rollouts=hs_rollouts,
            num_bins=int(args.bucket_num_bins),
            min_points_per_bin=int(args.bucket_min_points_per_bin),
            prefix_key_mode=str(args.bucket_prefix_key_mode),
        )

        response_ids: list[int] = []
        entropy_seq: list[float] = []
        step_lps_seq: list[dict[int, float]] = []
        step_records: list[dict[str, Any]] = []

        # Pass-1: generate one greedy trajectory and cache per-step logprobs.
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
            response_ids.append(int(greedy_token))
            entropy_seq.append(float(entropy_t))
            step_lps_seq.append(dict(step_lp))
            if tokenizer.eos_token_id is not None and int(greedy_token) == int(tokenizer.eos_token_id):
                break

        # Precompute realized suffix proxy F_t from greedy trajectory itself.
        n_steps = len(entropy_seq)
        if n_steps > 0:
            arr = np.array(entropy_seq, dtype=np.float64)
            suffix_sum = np.cumsum(arr[::-1])[::-1]
        else:
            suffix_sum = np.array([], dtype=np.float64)

        eval_positions: list[int] = []
        branch_count = 0
        for step_idx in range(n_steps):
            step_lp = step_lps_seq[step_idx]
            cands, _cand_probs = _topp_capped_from_step_logprobs(
                step_lp,
                top_p=float(args.candidate_top_p),
                max_k=int(args.candidate_max_k),
            )
            if (
                float(entropy_seq[step_idx]) >= float(args.entropy_threshold)
                and len(cands) >= 2
                and (int(args.max_branch_steps) <= 0 or branch_count < int(args.max_branch_steps))
                and (step_idx + 1) < n_steps
            ):
                eval_positions.append(step_idx)
                branch_count += 1

        for step_idx in eval_positions:
            step_lp = step_lps_seq[step_idx]
            cands, cand_probs = _topp_capped_from_step_logprobs(
                step_lp,
                top_p=float(args.candidate_top_p),
                max_k=int(args.candidate_max_k),
            )
            chosen_token = int(response_ids[step_idx])
            if chosen_token not in cands:
                summary_cnt["n_skipped_chosen_not_in_candidates"] += 1
                continue

            remaining = int(args.max_new_tokens) - step_idx - 1
            if remaining <= 0:
                continue
            prefixes = [prompt_ids + response_ids[:step_idx] + [int(t)] for t in cands]
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
            if len(f_mc) != len(cands):
                continue

            chosen_idx = cands.index(chosen_token)
            fbar_mc = float(sum(float(p) * float(v) for p, v in zip(cand_probs, f_mc, strict=False)))
            f_real_mc = float(f_mc[chosen_idx])
            sign_mc_real = _sign(fbar_mc - f_real_mc)

            pref_sum = float(np.sum(np.array(entropy_seq[: step_idx + 1], dtype=np.float64)))
            if str(args.bucket_prefix_key_mode) == "sum":
                pref_key = pref_sum
            else:
                pref_key = pref_sum / float(step_idx + 1)
            fbar_bucket = _bucket_lookup(bucket_estimator, pref_key)

            future_sum = float(suffix_sum[step_idx + 1])
            future_len = int(n_steps - step_idx - 1)
            if future_len <= 0:
                continue
            if str(args.bias_metrics_mode) == "length_normalized":
                f_real_proxy = future_sum / float(future_len)
            else:
                f_real_proxy = future_sum
            sign_bucket_real = _sign(float(fbar_bucket) - float(f_real_proxy))

            summary_cnt["n_eval_steps"] += 1
            summary_cnt["n_match_steps"] += int(sign_bucket_real == sign_mc_real)
            summary_cnt["n_match_steps_if_flip"] += int((-sign_bucket_real) == sign_mc_real)
            if sign_mc_real != 0:
                summary_cnt["n_mc_nonzero_steps"] += 1

            if bool(args.save_traces):
                step_records.append(
                    {
                        "step_index": int(step_idx),
                        "entropy_t": float(entropy_seq[step_idx]),
                        "chosen_token": int(chosen_token),
                        "chosen_text": tokenizer.decode([int(chosen_token)], skip_special_tokens=True),
                        "candidate_count": int(len(cands)),
                        "candidates": [int(x) for x in cands],
                        "candidate_probs_renorm_topp": [float(x) for x in cand_probs],
                        "f_mc_128": [float(x) for x in f_mc],
                        "f_bar_mc_128": float(fbar_mc),
                        "f_real_mc_128": float(f_real_mc),
                        "sign_real_mc_128": int(sign_mc_real),
                        "bucket_prefix_key_mode": str(args.bucket_prefix_key_mode),
                        "bucket_prefix_key": float(pref_key),
                        "f_bar_bucket_proxy": float(fbar_bucket),
                        "f_real_proxy_from_trajectory": float(f_real_proxy),
                        "sign_real_bucket_proxy": int(sign_bucket_real),
                        "sign_match_real": bool(sign_bucket_real == sign_mc_real),
                        "sign_match_real_if_flip": bool((-sign_bucket_real) == sign_mc_real),
                    }
                )

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
            "bucket_prefix_key_mode": str(args.bucket_prefix_key_mode),
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
            "real_sign_match_acc_all": _mean_ratio(int(total["n_match_steps"]), int(total["n_eval_steps"])),
            "real_sign_match_acc_mc_nonzero_only": _mean_ratio(
                int(total["n_match_steps"]), int(total["n_mc_nonzero_steps"])
            ),
            "real_sign_match_acc_if_flip_all": _mean_ratio(
                int(total["n_match_steps_if_flip"]), int(total["n_eval_steps"])
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
                "bucket_prefix_key_mode": str(args.bucket_prefix_key_mode),
            },
        }
        with open(out_dir / "sign_compare_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()

