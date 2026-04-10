#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
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
    *,
    m_iter: Any | None = None,
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
    loop = m_iter if m_iter is not None else range(m_samples)
    for _ in loop:
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
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--top_entropy_ratio", type=float, default=0.10)
    parser.add_argument("--max_positions_per_rollout", type=int, default=20)
    parser.add_argument(
        "--mc_m_samples",
        type=int,
        default=64,
        help="MC samples per F estimate (suffix entropy sum); higher = lower variance.",
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
    parser.add_argument("--vllm_shard_rank", type=int, default=None)
    parser.add_argument("--vllm_shard_world_size", type=int, default=None)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--progress_all_ranks",
        action="store_true",
        help="Show tqdm on every shard process (default: rank 0 only).",
    )
    parser.add_argument(
        "--progress_nested",
        action="store_true",
        help="Nested tqdm: rollouts → positions → MC samples per F estimate.",
    )
    args = parser.parse_args()
    if not (0.0 < float(args.candidate_top_p) <= 1.0):
        raise SystemExit("--candidate_top_p must be in (0, 1].")
    if int(args.candidate_max_k) < 1:
        raise SystemExit("--candidate_max_k must be >= 1.")
    if args.candidate_mode == "fixed" and int(args.topk_alt) < 1:
        raise SystemExit("--topk_alt must be >= 1 when candidate_mode=fixed.")

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

    diag: dict[str, int] = {
        "n_prompts": 0,
        "n_empty_ground_truth": 0,
        "n_skip_all_rollouts_empty": 0,
        "n_skip_no_mixed": 0,
        "n_mixed_pairs_ok": 0,
        "n_skip_empty_step_lp": 0,
        "n_jsonl_lines": 0,
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
    _tqdm_kw = {"file": sys.stderr, "dynamic_ncols": True}

    prompt_iter = enumerate(local_rows)
    if use_tqdm:
        prompt_iter = tqdm(
            prompt_iter,
            total=len(local_rows),
            desc=f"{shard_desc} prompts",
            position=bar_base,
            leave=True,
            mininterval=0.5,
            **_tqdm_kw,
        )

    for local_i, row in prompt_iter:
        diag["n_prompts"] += 1
        global_idx = local_i * world_size + rank
        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        data_source = row.get("data_source", "math_dapo")
        ground_truth = _ground_truth_from_row(row)
        if not str(ground_truth).strip():
            diag["n_empty_ground_truth"] += 1

        rollouts: list[dict[str, Any]] = []
        rollout_iter = range(args.rollouts_per_prompt)
        if use_nested and tqdm is not None:
            rollout_iter = tqdm(
                rollout_iter,
                total=args.rollouts_per_prompt,
                desc=f"{shard_desc} rollouts",
                position=bar_base + 1,
                leave=False,
                mininterval=0.2,
                **_tqdm_kw,
            )
        for rollout_idx in rollout_iter:
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
            positions = pick_top_entropy_positions(entropies, args.top_entropy_ratio, args.max_positions_per_rollout)
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

                def _mc_iter() -> Any:
                    if use_nested and tqdm is not None:
                        return tqdm(
                            range(args.mc_m_samples),
                            total=args.mc_m_samples,
                            desc=f"{shard_desc} MC",
                            position=bar_base + 3,
                            leave=False,
                            mininterval=0.05,
                            **_tqdm_kw,
                        )
                    return range(args.mc_m_samples)

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
                    m_iter=_mc_iter(),
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
                        m_iter=_mc_iter(),
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
                    "candidate_mode": str(args.candidate_mode),
                    "alt_k_topp": int(cand_meta["k_topp"]),
                    "alt_k_used": int(cand_meta["k_used"]),
                    "topk_alt_token_ids": [int(t) for t in tids],
                    "topk_alt_probs": [float(p) for p in probs],
                }
                with open(part_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    diag["n_jsonl_lines"] += 1

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

