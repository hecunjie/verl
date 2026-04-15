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


def _build_rollout_prefix_suffix_refs(hs_rollouts: list[list[float]]) -> list[dict[str, Any]]:
    """Per-rollout references keyed by monotonic prefix entropy sum.

    For each rollout and each split position u in [1, n-1], store:
      - prefix_sum(u) = sum_{s<=u} h_s
      - suffix_sum(u) = sum_{s>u} h_s
      - suffix_rate(u) = suffix_sum(u) / (n-u)
    """
    refs: list[dict[str, Any]] = []
    for hs in hs_rollouts:
        n = len(hs)
        if n < 2:
            continue
        arr = np.array(hs, dtype=np.float64)
        pref = np.cumsum(arr)[:-1]
        suf = np.cumsum(arr[::-1])[::-1][1:]
        den = np.arange(n - 1, 0, -1, dtype=np.float64)  # n-u for u=1..n-1
        rate = suf / den
        refs.append(
            {
                "prefix_sum": pref,
                "suffix_sum": suf,
                "suffix_rate": rate,
            }
        )
    return refs


def _nearest_idx_on_monotonic_prefix(prefix_sum_arr: np.ndarray, target: float) -> int:
    if prefix_sum_arr.size == 0:
        return -1
    idx = int(np.searchsorted(prefix_sum_arr, target, side="left"))
    if idx <= 0:
        return 0
    if idx >= int(prefix_sum_arr.size):
        return int(prefix_sum_arr.size - 1)
    left = idx - 1
    right = idx
    dl = abs(float(prefix_sum_arr[left]) - float(target))
    dr = abs(float(prefix_sum_arr[right]) - float(target))
    # Tie-break toward earlier position (left), matching "from head to tail" intuition.
    return left if dl <= dr else right


def _query_fbar_from_refs_by_prefix_sum(refs: list[dict[str, Any]], prefix_sum_target: float) -> tuple[float, float, int]:
    """Return mean suffix_sum / suffix_rate across rollouts at nearest prefix-sum positions."""
    picked_sum: list[float] = []
    picked_rate: list[float] = []
    for r in refs:
        p = r.get("prefix_sum")
        ssum = r.get("suffix_sum")
        srate = r.get("suffix_rate")
        if not isinstance(p, np.ndarray) or not isinstance(ssum, np.ndarray) or not isinstance(srate, np.ndarray):
            continue
        i = _nearest_idx_on_monotonic_prefix(p, float(prefix_sum_target))
        if i < 0:
            continue
        picked_sum.append(float(ssum[i]))
        picked_rate.append(float(srate[i]))
    if not picked_sum:
        return 0.0, 0.0, 0
    return float(np.mean(np.array(picked_sum, dtype=np.float64))), float(np.mean(np.array(picked_rate, dtype=np.float64))), int(len(picked_sum))


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
        choices=["sum", "rate", "both"],
        default="both",
        help="Which proxy metric to report. Retrieval key is always prefix entropy sum.",
    )

    parser.add_argument("--save_traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--math_eval_backend",
        choices=["auto", "math_dapo", "math_verify"],
        default="auto",
        help="Math correctness backend for math-like datasets.",
    )
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

    mode_cfg = str(args.bucket_prefix_key_mode)
    eval_modes = ["sum", "rate"] if mode_cfg == "both" else [mode_cfg]

    summary_cnt: dict[str, int] = {
        "n_prompts": 0,
        "n_eval_steps": 0,
        "n_mc_nonzero_steps": 0,
        "n_match_steps_sum": 0,
        "n_match_steps_if_flip_sum": 0,
        "n_match_steps_rate": 0,
        "n_match_steps_if_flip_rate": 0,
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
        rollout_refs = _build_rollout_prefix_suffix_refs(hs_rollouts)

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

            future_sum = float(suffix_sum[step_idx + 1])
            future_len = int(n_steps - step_idx - 1)
            if future_len <= 0:
                continue
            if str(args.bias_metrics_mode) == "length_normalized":
                f_real_proxy = future_sum / float(future_len)
            else:
                f_real_proxy = future_sum
            fbar_bucket_sum, fbar_bucket_rate, n_ref_hits = _query_fbar_from_refs_by_prefix_sum(
                rollout_refs, pref_sum
            )
            sign_bucket_real_sum = _sign(float(fbar_bucket_sum) - float(f_real_proxy))
            sign_bucket_real_rate = _sign(float(fbar_bucket_rate) - float(f_real_proxy))

            summary_cnt["n_eval_steps"] += 1
            if "sum" in eval_modes:
                summary_cnt["n_match_steps_sum"] += int(sign_bucket_real_sum == sign_mc_real)
                summary_cnt["n_match_steps_if_flip_sum"] += int((-sign_bucket_real_sum) == sign_mc_real)
            if "rate" in eval_modes:
                summary_cnt["n_match_steps_rate"] += int(sign_bucket_real_rate == sign_mc_real)
                summary_cnt["n_match_steps_if_flip_rate"] += int((-sign_bucket_real_rate) == sign_mc_real)
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
                        "bucket_prefix_key_sum": float(pref_sum),
                        "bucket_ref_hits": int(n_ref_hits),
                        "f_bar_bucket_proxy_sum": float(fbar_bucket_sum),
                        "f_bar_bucket_proxy_rate": float(fbar_bucket_rate),
                        "f_real_proxy_from_trajectory": float(f_real_proxy),
                        "sign_real_bucket_proxy_sum": int(sign_bucket_real_sum),
                        "sign_real_bucket_proxy_rate": int(sign_bucket_real_rate),
                        "sign_match_real_sum": bool(sign_bucket_real_sum == sign_mc_real),
                        "sign_match_real_rate": bool(sign_bucket_real_rate == sign_mc_real),
                        "sign_match_real_if_flip_sum": bool((-sign_bucket_real_sum) == sign_mc_real),
                        "sign_match_real_if_flip_rate": bool((-sign_bucket_real_rate) == sign_mc_real),
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
            "real_sign_match_acc_all_sum": (
                _mean_ratio(int(total["n_match_steps_sum"]), int(total["n_eval_steps"])) if "sum" in eval_modes else float("nan")
            ),
            "real_sign_match_acc_mc_nonzero_only_sum": (
                _mean_ratio(int(total["n_match_steps_sum"]), int(total["n_mc_nonzero_steps"]))
                if "sum" in eval_modes
                else float("nan")
            ),
            "real_sign_match_acc_if_flip_all_sum": (
                _mean_ratio(int(total["n_match_steps_if_flip_sum"]), int(total["n_eval_steps"]))
                if "sum" in eval_modes
                else float("nan")
            ),
            "real_sign_match_acc_all_rate": (
                _mean_ratio(int(total["n_match_steps_rate"]), int(total["n_eval_steps"]))
                if "rate" in eval_modes
                else float("nan")
            ),
            "real_sign_match_acc_mc_nonzero_only_rate": (
                _mean_ratio(int(total["n_match_steps_rate"]), int(total["n_mc_nonzero_steps"]))
                if "rate" in eval_modes
                else float("nan")
            ),
            "real_sign_match_acc_if_flip_all_rate": (
                _mean_ratio(int(total["n_match_steps_if_flip_rate"]), int(total["n_eval_steps"]))
                if "rate" in eval_modes
                else float("nan")
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

