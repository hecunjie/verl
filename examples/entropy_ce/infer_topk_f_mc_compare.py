#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
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
    clamp_vllm_logprobs_topk,
    entropy_from_logprobs_topk,
    estimate_F_mc_many_prefixes_vllm,
    evaluate_solution_acc,
    file_sync,
    init_dist,
    load_data,
    purge_all_torchrun_like_env_for_vllm_standalone,
    vllm_generate_quiet,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _ground_truth_from_row(row: dict[str, Any]) -> str:
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


def _is_math_like_source(data_source: str) -> bool:
    ds = str(data_source or "")
    dsl = ds.lower()
    return (
        ds in {
            "math_dapo",
            "math",
            "math_dapo_reasoning",
            "lighteval/MATH",
            "DigitalLearningGmbH/MATH-lighteval",
            "HuggingFaceH4/MATH-500",
        }
        or ds.startswith("aime")
        or ("math500" in dsl)
    )


def _append_boxed_instruction(prompt_text: str) -> str:
    suffix = (
        "\n\nPlease end your final answer with exactly one LaTeX boxed form: "
        "\\boxed{...}."
    )
    if "\\boxed" in prompt_text:
        return prompt_text
    return f"{prompt_text.rstrip()}{suffix}"


def _extract_last_boxed_content(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} occurrence."""
    if not text:
        return None
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    for st in reversed(starts):
        brace_start = text.find("{", st)
        if brace_start < 0:
            continue
        depth = 0
        for i in range(brace_start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    inner = text[brace_start + 1 : i].strip()
                    return inner if inner else None
    return None


def _evaluate_final_boxed_only(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: str,
    math_eval_backend: str,
) -> tuple[bool, dict[str, Any], str | None]:
    """Strict evaluator: only score the last boxed answer."""
    boxed = _extract_last_boxed_content(solution_str)
    if boxed is None:
        return False, {"mode": "final_boxed_only", "reason": "no_boxed_found"}, None
    boxed_solution = f"Answer: \\boxed{{{boxed}}}"
    ok, eval_info = evaluate_solution_acc(
        data_source=data_source,
        solution_str=boxed_solution,
        ground_truth=ground_truth,
        math_eval_backend=math_eval_backend,
    )
    return bool(ok), {"mode": "final_boxed_only", "inner_eval": eval_info}, boxed


def _step_logprobs_vllm(llm: Any, prefix_ids: list[int], logprobs_k: int) -> tuple[int | None, dict[int, float]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    k = clamp_vllm_logprobs_topk(logprobs_k)
    sp = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, logprobs=k)
    out = vllm_generate_quiet(llm, [TokensPrompt(prompt_token_ids=prefix_ids)], sp)
    o = out[0].outputs[0]
    if not o.token_ids:
        return None, {}
    token_id = int(o.token_ids[0])
    if not o.logprobs:
        return token_id, {}
    step_lp_raw = o.logprobs[0]
    step_lp: dict[int, float] = {}
    for tid, info in step_lp_raw.items():
        step_lp[int(tid)] = float(info.logprob)
    return token_id, step_lp


def _sample_one_token_vllm(
    *,
    llm: Any,
    prefix_ids: list[int],
    temperature: float,
    top_p: float,
) -> int | None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(
        max_tokens=1,
        temperature=float(temperature),
        top_p=float(top_p),
        logprobs=0,
    )
    out = vllm_generate_quiet(llm, [TokensPrompt(prompt_token_ids=prefix_ids)], sp)
    o = out[0].outputs[0]
    if not o.token_ids:
        return None
    return int(o.token_ids[0])


def _candidate_lookahead_1step_entropy(
    llm: Any,
    prefix_ids: list[int],
    candidate_token_id: int,
    logprobs_k: int,
) -> float:
    """Use next-step entropy after appending one candidate token as F proxy."""
    _, step_lp = _step_logprobs_vllm(
        llm=llm,
        prefix_ids=prefix_ids + [int(candidate_token_id)],
        logprobs_k=logprobs_k,
    )
    return float(entropy_from_logprobs_topk(step_lp))


def _sample_group_rollouts_for_bucket(
    *,
    llm: Any,
    prompt_ids: list[int],
    n_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logprobs_k: int,
    batch_chunk: int,
) -> list[list[float]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    k = clamp_vllm_logprobs_topk(logprobs_k)
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=k,
    )
    out_h: list[list[float]] = []
    for start in range(0, int(n_rollouts), int(batch_chunk)):
        bs = min(int(batch_chunk), int(n_rollouts) - start)
        prompts = [TokensPrompt(prompt_token_ids=prompt_ids)] * bs
        outputs = vllm_generate_quiet(llm, prompts, sp)
        if len(outputs) != bs:
            raise RuntimeError(f"vLLM batch size mismatch: expected {bs}, got {len(outputs)}")
        for out_req in outputs:
            o = out_req.outputs[0]
            hs: list[float] = []
            for step_lp in o.logprobs or []:
                d: dict[int, float] = {}
                for tid, info in step_lp.items():
                    d[int(tid)] = float(info.logprob)
                hs.append(float(entropy_from_logprobs_topk(d)))
            if hs:
                out_h.append(hs)
    return out_h


def _build_bucket_group_estimator(
    *,
    llm: Any,
    prompt_ids: list[int],
    n_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logprobs_k: int,
    batch_chunk: int,
    num_bins: int,
    min_points_per_bin: int,
) -> dict[str, Any] | None:
    b = max(2, int(num_bins))

    def _make_padded_degenerate(global_mean: float) -> dict[str, Any]:
        # Keep fixed-size bucket payload so downstream always sees exactly b bins.
        return {
            "degenerate": True,
            "edges": [0.0 for _ in range(b + 1)],
            "means": [float(global_mean) for _ in range(b)],
            "valid": [True for _ in range(b)],
            "global_mean_suffix_rate": float(global_mean),
        }

    hs_rollouts = _sample_group_rollouts_for_bucket(
        llm=llm,
        prompt_ids=prompt_ids,
        n_rollouts=n_rollouts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs_k=logprobs_k,
        batch_chunk=batch_chunk,
    )
    prefix_rates: list[float] = []
    suffix_rates: list[float] = []
    for hs in hs_rollouts:
        n = len(hs)
        if n < 2:
            continue
        suffix_sum = np.cumsum(np.array(hs[::-1], dtype=np.float64))[::-1]
        pref_sum = 0.0
        for u in range(1, n):
            pref_sum += float(hs[u - 1])
            pref_rate = pref_sum / float(u)
            suffix_rate = float(suffix_sum[u] / float(n - u))
            prefix_rates.append(pref_rate)
            suffix_rates.append(suffix_rate)
    if not prefix_rates:
        return _make_padded_degenerate(global_mean=0.0)

    p = np.array(prefix_rates, dtype=np.float64)
    f = np.array(suffix_rates, dtype=np.float64)
    pmin = float(np.min(p))
    pmax = float(np.max(p))
    if not np.isfinite(pmin) or not np.isfinite(pmax):
        return _make_padded_degenerate(global_mean=float(np.mean(f)))
    if abs(pmax - pmin) < 1e-12:
        return _make_padded_degenerate(global_mean=float(np.mean(f)))

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
    global_mean = float(np.mean(f))
    return {
        "degenerate": False,
        "edges": [float(x) for x in edges.tolist()],
        "means": [float(x) for x in means.tolist()],
        "valid": [bool(x) for x in valid.tolist()],
        "global_mean_suffix_rate": float(global_mean),
    }


def _bucket_estimate_bar_f(estimator: dict[str, Any] | None, prefix_rate: float) -> float:
    if estimator is None:
        return 0.0
    gm = float(estimator.get("global_mean_suffix_rate", 0.0))
    if bool(estimator.get("degenerate", False)):
        return gm
    edges = estimator.get("edges")
    means = estimator.get("means")
    valid = estimator.get("valid")
    if not isinstance(edges, list) or not isinstance(means, list) or not isinstance(valid, list):
        return gm
    b = len(means)
    if b == 0:
        return gm
    idx = int(np.searchsorted(np.array(edges, dtype=np.float64), float(prefix_rate), side="right") - 1)
    idx = min(max(idx, 0), b - 1)
    if valid[idx]:
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


def _topk_from_step_logprobs(step_logprobs: dict[int, float], topk: int) -> tuple[list[int], list[float]]:
    if not step_logprobs:
        return [], []
    items = sorted(step_logprobs.items(), key=lambda x: x[1], reverse=True)[: max(1, topk)]
    tids = [int(tid) for tid, _ in items]
    lps = np.array([float(lp) for _, lp in items], dtype=np.float64)
    m = float(np.max(lps))
    p = np.exp(np.clip(lps - m, -80.0, 0.0))
    z = float(np.sum(p))
    if z <= 0.0 or not np.isfinite(z):
        probs = np.ones(len(tids), dtype=np.float64) / float(len(tids))
    else:
        probs = p / z
    return tids, [float(x) for x in probs.tolist()]


def _topp_capped_from_step_logprobs(
    step_logprobs: dict[int, float],
    top_p: float,
    max_k: int,
) -> tuple[list[int], list[float]]:
    if not step_logprobs:
        return [], []
    items = sorted(step_logprobs.items(), key=lambda x: x[1], reverse=True)
    tids_all = [int(tid) for tid, _ in items]
    lps = np.array([float(lp) for _, lp in items], dtype=np.float64)
    m = float(np.max(lps))
    p = np.exp(np.clip(lps - m, -80.0, 0.0))
    z = float(np.sum(p))
    if z <= 0.0 or not np.isfinite(z):
        p = np.ones(len(tids_all), dtype=np.float64) / float(len(tids_all))
    else:
        p = p / z
    tp = float(np.clip(top_p, 1e-12, 1.0))
    cum = np.cumsum(p)
    idx = int(np.searchsorted(cum, tp, side="left"))
    k_topp = max(1, idx + 1)
    k_used = min(max(1, int(max_k)), k_topp)
    sel_tids = tids_all[:k_used]
    sel_p = p[:k_used]
    z2 = float(np.sum(sel_p))
    if z2 <= 0.0 or not np.isfinite(z2):
        sel_p = np.ones(len(sel_tids), dtype=np.float64) / float(len(sel_tids))
    else:
        sel_p = sel_p / z2
    return sel_tids, [float(x) for x in sel_p.tolist()]


def _decode_one_policy(
    *,
    llm: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    policy: str,
    entropy_threshold: float,
    candidate_top_p: float,
    candidate_max_k: int,
    selection_f_mode: str,
    max_new_tokens: int,
    mc_m_samples: int,
    mc_temperature: float,
    mc_top_p: float,
    sampling_temperature: float,
    sampling_top_p: float,
    vllm_logprobs_topk: int,
    vllm_request_batch_chunk: int,
    f_continuation_mode: str,
    f_sentence_max_new_tokens: int,
    f_sentence_stop: str,
    normalize_by_continuation_length: bool,
    max_branch_steps: int,
    rng: random.Random,
    eos_token_id: int | None,
    show_nested_progress: bool = False,
    progress_position: int = 1,
    progress_desc: str = "decode",
    progress_mc_position: int = 2,
    progress_mc_desc: str = "mc",
    bucket_group_estimator: dict[str, Any] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    if policy not in {"min_f_mc", "sampling_baseline", "greedy_baseline"}:
        raise ValueError(f"unsupported policy: {policy}")

    response_ids: list[int] = []
    branch_records: list[dict[str, Any]] = []
    branch_count = 0
    entropy_hist: list[float] = []
    sentence_stop_check = None
    if f_sentence_stop == "pysbd":
        from sentence_stop_utils import make_pysbd_first_sentence_stop_check

        sentence_stop_check = make_pysbd_first_sentence_stop_check()

    step_iter = range(max_new_tokens)
    if show_nested_progress and tqdm is not None:
        step_iter = tqdm(
            step_iter,
            total=max_new_tokens,
            desc=progress_desc,
            dynamic_ncols=True,
            position=progress_position,
            leave=False,
        )

    for step_idx in step_iter:
        prefix = prompt_ids + response_ids
        if policy == "sampling_baseline":
            sampled_token = _sample_one_token_vllm(
                llm=llm,
                prefix_ids=prefix,
                temperature=sampling_temperature,
                top_p=sampling_top_p,
            )
            if sampled_token is None:
                break
            chosen_token = int(sampled_token)
            response_ids.append(chosen_token)
            entropy_hist.append(float("nan"))
            if eos_token_id is not None and chosen_token == int(eos_token_id):
                break
            continue

        greedy_token, step_lp = _step_logprobs_vllm(
            llm=llm,
            prefix_ids=prefix,
            logprobs_k=max(vllm_logprobs_topk, candidate_max_k),
        )
        if greedy_token is None:
            break

        entropy_t = entropy_from_logprobs_topk(step_lp)
        cands, cand_probs = _topp_capped_from_step_logprobs(
            step_lp,
            top_p=candidate_top_p,
            max_k=candidate_max_k,
        )
        if not cands:
            cands = [greedy_token]
            cand_probs = [1.0]

        should_branch = (
            policy == "min_f_mc"
            and entropy_t >= entropy_threshold
            and len(cands) >= 2
            and (max_branch_steps <= 0 or branch_count < max_branch_steps)
        )

        chosen_token = greedy_token
        f_values: list[float] = []
        if should_branch:
            if policy == "min_f_mc":
                remaining = max_new_tokens - step_idx - 1
                if remaining <= 0:
                    chosen_token = int(cands[0])
                else:
                    prefixes = [(prompt_ids + response_ids + [int(t)]) for t in cands]
                    if selection_f_mode == "greedy_path":
                        m_samples_use = 1
                        temp_use = 0.0
                        top_p_use = 1.0
                    elif selection_f_mode == "mc":
                        m_samples_use = int(mc_m_samples)
                        temp_use = float(mc_temperature)
                        top_p_use = float(mc_top_p)
                    elif selection_f_mode == "lookahead_1step":
                        f_values = [
                            _candidate_lookahead_1step_entropy(
                                llm=llm,
                                prefix_ids=(prompt_ids + response_ids),
                                candidate_token_id=int(t),
                                logprobs_k=vllm_logprobs_topk,
                            )
                            for t in cands
                        ]
                        best_i = int(np.argmin(np.array(f_values, dtype=np.float64)))
                        chosen_token = int(cands[best_i])
                        m_samples_use = 0
                        temp_use = 0.0
                        top_p_use = 1.0
                    elif selection_f_mode == "bucket_group_estimate":
                        prefix_sum = float(sum(entropy_hist) + float(entropy_t))
                        prefix_len = len(entropy_hist) + 1
                        f_values = []
                        for t in cands:
                            h1 = _candidate_lookahead_1step_entropy(
                                llm=llm,
                                prefix_ids=(prompt_ids + response_ids),
                                candidate_token_id=int(t),
                                logprobs_k=vllm_logprobs_topk,
                            )
                            p_rate = (prefix_sum + float(h1)) / float(prefix_len + 1)
                            f_values.append(float(_bucket_estimate_bar_f(bucket_group_estimator, p_rate)))
                        best_i = int(np.argmin(np.array(f_values, dtype=np.float64)))
                        chosen_token = int(cands[best_i])
                        m_samples_use = 0
                        temp_use = 0.0
                        top_p_use = 1.0
                    else:
                        raise ValueError(f"unsupported selection_f_mode: {selection_f_mode}")

                    if selection_f_mode in ("greedy_path", "mc"):
                        n_req = len(prefixes) * int(m_samples_use)
                        bs = int(vllm_request_batch_chunk)
                        chunk_starts = range(0, n_req, bs)
                        if show_nested_progress and tqdm is not None:
                            n_chunks = (n_req + bs - 1) // bs
                            chunk_starts = tqdm(
                                chunk_starts,
                                total=n_chunks,
                                desc=progress_mc_desc,
                                dynamic_ncols=True,
                                position=progress_mc_position,
                                leave=False,
                            )
                        f_values = estimate_F_mc_many_prefixes_vllm(
                            llm=llm,
                            prefixes=prefixes,
                            m_samples=m_samples_use,
                            max_new_tokens=remaining,
                            temperature=temp_use,
                            top_p=top_p_use,
                            logprobs_k=vllm_logprobs_topk,
                            batch_chunk=vllm_request_batch_chunk,
                            f_continuation_mode=f_continuation_mode,
                            tokenizer=tokenizer if f_continuation_mode == "first_sentence" else None,
                            f_sentence_max_new_tokens=f_sentence_max_new_tokens,
                            sentence_stop_check=sentence_stop_check,
                            normalize_by_continuation_length=normalize_by_continuation_length,
                            chunk_starts_iter=chunk_starts,
                        )
                        if not f_values or len(f_values) != len(cands):
                            chosen_token = int(cands[0])
                        else:
                            best_i = int(np.argmin(np.array(f_values, dtype=np.float64)))
                            chosen_token = int(cands[best_i])
            else:
                raise ValueError(f"unsupported policy: {policy}")

            branch_count += 1
            branch_records.append(
                {
                    "step_index": int(step_idx),
                    "entropy_t": float(entropy_t),
                    "candidates": [int(x) for x in cands],
                    "candidate_probs_renorm_topp": [float(x) for x in cand_probs],
                    "candidate_texts": [tokenizer.decode([int(x)], skip_special_tokens=True) for x in cands],
                    "f_values_mc": [float(x) for x in f_values] if f_values else None,
                    "selection_f_mode": selection_f_mode if policy == "min_f_mc" else None,
                    "chosen_token": int(chosen_token),
                    "chosen_text": tokenizer.decode([int(chosen_token)], skip_special_tokens=True),
                }
            )

        response_ids.append(int(chosen_token))
        entropy_hist.append(float(entropy_t))
        if eos_token_id is not None and int(chosen_token) == int(eos_token_id):
            break

    return response_ids, branch_records


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare min-F_mc decoding vs pure sampling baseline and greedy baseline."
    )
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    parser.add_argument("--candidate_top_p", type=float, default=0.95)
    parser.add_argument("--candidate_max_k", type=int, default=5)
    parser.add_argument(
        "--selection_f_mode",
        choices=["mc", "greedy_path", "lookahead_1step", "bucket_group_estimate"],
        default="greedy_path",
        help="How to estimate candidate F when selecting min-F: mc (multi-sample), greedy_path (single deterministic continuation), lookahead_1step (H_{t+1} proxy), or bucket_group_estimate (group bucket-aligned estimate).",
    )
    parser.add_argument("--max_branch_steps", type=int, default=0, help="<=0 means no cap.")

    parser.add_argument("--mc_m_samples", type=int, default=1)
    parser.add_argument("--mc_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=0.95)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_top_p", type=float, default=0.95)
    parser.add_argument("--bias_metrics_mode", choices=["raw", "length_normalized"], default="length_normalized")
    parser.add_argument("--f_continuation_mode", choices=["full", "first_sentence"], default="first_sentence")
    parser.add_argument("--f_sentence_max_new_tokens", type=int, default=256)
    parser.add_argument("--f_sentence_stop", choices=["simple", "pysbd"], default="simple")

    parser.add_argument("--vllm_logprobs_topk", type=int, default=20)
    parser.add_argument("--vllm_request_batch_chunk", type=int, default=64)
    parser.add_argument("--bucket_group_rollouts", type=int, default=16)
    parser.add_argument("--bucket_num_bins", type=int, default=100)
    parser.add_argument("--bucket_min_points_per_bin", type=int, default=4)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--save_traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--progress_all_ranks",
        action="store_true",
        help="Show progress bars on all ranks (default: rank0 only).",
    )
    parser.add_argument(
        "--progress_nested",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show nested decode/MC progress bars per sample.",
    )
    parser.add_argument(
        "--progress_echo",
        action="store_true",
        help="Rank0 prints prompt-level elapsed logs to stderr.",
    )

    parser.add_argument("--vllm_shard_rank", type=int, default=None)
    parser.add_argument("--vllm_shard_world_size", type=int, default=None)
    parser.add_argument(
        "--math_eval_backend",
        choices=["auto", "math_dapo", "math_verify"],
        default="auto",
        help="Math correctness backend for math-like datasets; use math_verify for MATH-500 style evaluation.",
    )
    parser.add_argument(
        "--force_boxed_answer_instruction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For math-like data_source, append an explicit instruction requiring final answer in \\boxed{...}.",
    )
    args = parser.parse_args()

    if not (0.0 < float(args.candidate_top_p) <= 1.0):
        raise SystemExit("--candidate_top_p must be in (0, 1].")
    if int(args.candidate_max_k) < 2:
        raise SystemExit("--candidate_max_k should be >=2 for meaningful compare.")
    if args.mc_m_samples < 1:
        raise SystemExit("--mc_m_samples must be >=1.")
    if float(args.sampling_temperature) < 0.0:
        raise SystemExit("--sampling_temperature must be >=0.")
    if not (0.0 < float(args.sampling_top_p) <= 1.0):
        raise SystemExit("--sampling_top_p must be in (0, 1].")
    if args.max_new_tokens < 1:
        raise SystemExit("--max_new_tokens must be >=1.")
    if args.vllm_request_batch_chunk < 1:
        raise SystemExit("--vllm_request_batch_chunk must be >=1.")
    if args.bucket_group_rollouts < 1:
        raise SystemExit("--bucket_group_rollouts must be >=1.")
    if args.bucket_num_bins < 2:
        raise SystemExit("--bucket_num_bins must be >=2.")
    if args.bucket_min_points_per_bin < 1:
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
    # Clear stale file-sync markers from previous runs in the same output dir.
    # Otherwise rank0 may observe old "_done_infer_compare_rank*" files and pass barrier early.
    if rank == 0:
        for r in range(world_size):
            stale = out_dir / f"_done_infer_compare_rank{r}"
            if stale.exists():
                stale.unlink()
    part_path = out_dir / f"infer_compare_rank{rank}.jsonl"
    part_path.write_text("", encoding="utf-8")

    rows = load_data(args.input_data, args.max_samples, args.seed)
    local_rows = [row for idx, row in enumerate(rows) if idx % world_size == rank]

    rng = random.Random(args.seed + rank)
    np.random.seed(args.seed + rank)

    env_no_tqdm = os.environ.get("TQDM_DISABLE", "").strip().lower() in ("1", "true", "yes")
    use_tqdm = tqdm is not None and not args.no_progress and not env_no_tqdm and (
        args.progress_all_ranks or rank == 0
    )
    use_nested = bool(use_tqdm and args.progress_nested)
    bar_base = (rank * 5) if (use_tqdm and use_nested and args.progress_all_ranks) else (rank if use_tqdm else 0)

    acc_minf: list[float] = []
    acc_rand: list[float] = []
    acc_greedy: list[float] = []
    acc_minf_boxed: list[float] = []
    acc_rand_boxed: list[float] = []
    acc_greedy_boxed: list[float] = []
    branch_count_minf: list[float] = []
    branch_count_rand: list[float] = []
    branch_count_greedy: list[float] = []
    token_len_minf: list[float] = []
    token_len_rand: list[float] = []
    token_len_greedy: list[float] = []

    prompt_iter = enumerate(local_rows)
    if use_tqdm:
        prompt_iter = tqdm(
            prompt_iter,
            total=len(local_rows),
            desc=f"shard{rank} prompts",
            dynamic_ncols=True,
            position=bar_base,
            leave=True,
        )

    run_t0 = 0.0
    if args.progress_echo and rank == 0:
        run_t0 = time.perf_counter()

    for local_i, row in prompt_iter:
        global_idx = local_i * world_size + rank
        if args.progress_echo and rank == 0:
            elapsed = time.perf_counter() - run_t0
            print(
                f"[infer_compare] rank0 prompt {local_i + 1}/{len(local_rows)} global#{global_idx} elapsed={elapsed:.1f}s",
                flush=True,
            )
        data_source = row.get("data_source", "math_dapo")
        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        if args.force_boxed_answer_instruction and _is_math_like_source(str(data_source)):
            prompt_text = _append_boxed_instruction(prompt_text)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        ground_truth = _ground_truth_from_row(row)
        bucket_estimator: dict[str, Any] | None = None
        if str(args.selection_f_mode) == "bucket_group_estimate":
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

        response_minf, trace_minf = _decode_one_policy(
            llm=llm,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            policy="min_f_mc",
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
            vllm_logprobs_topk=int(args.vllm_logprobs_topk),
            vllm_request_batch_chunk=int(args.vllm_request_batch_chunk),
            f_continuation_mode=str(args.f_continuation_mode),
            f_sentence_max_new_tokens=int(args.f_sentence_max_new_tokens),
            f_sentence_stop=str(args.f_sentence_stop),
            normalize_by_continuation_length=(args.bias_metrics_mode == "length_normalized"),
            max_branch_steps=int(args.max_branch_steps),
            rng=rng,
            eos_token_id=tokenizer.eos_token_id,
            show_nested_progress=bool(use_nested),
            progress_position=bar_base + 1,
            progress_desc=f"shard{rank} minF decode",
            progress_mc_position=bar_base + 2,
            progress_mc_desc=f"shard{rank} minF MC",
            bucket_group_estimator=bucket_estimator,
        )
        response_rand, trace_rand = _decode_one_policy(
            llm=llm,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            policy="sampling_baseline",
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
            vllm_logprobs_topk=int(args.vllm_logprobs_topk),
            vllm_request_batch_chunk=int(args.vllm_request_batch_chunk),
            f_continuation_mode=str(args.f_continuation_mode),
            f_sentence_max_new_tokens=int(args.f_sentence_max_new_tokens),
            f_sentence_stop=str(args.f_sentence_stop),
            normalize_by_continuation_length=(args.bias_metrics_mode == "length_normalized"),
            max_branch_steps=int(args.max_branch_steps),
            rng=rng,
            eos_token_id=tokenizer.eos_token_id,
            show_nested_progress=bool(use_nested),
            progress_position=bar_base + 3,
            progress_desc=f"shard{rank} sampling decode",
            progress_mc_position=bar_base + 2,
            progress_mc_desc=f"shard{rank} sampling MC",
            bucket_group_estimator=None,
        )
        response_greedy, trace_greedy = _decode_one_policy(
            llm=llm,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            policy="greedy_baseline",
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
            vllm_logprobs_topk=int(args.vllm_logprobs_topk),
            vllm_request_batch_chunk=int(args.vllm_request_batch_chunk),
            f_continuation_mode=str(args.f_continuation_mode),
            f_sentence_max_new_tokens=int(args.f_sentence_max_new_tokens),
            f_sentence_stop=str(args.f_sentence_stop),
            normalize_by_continuation_length=(args.bias_metrics_mode == "length_normalized"),
            max_branch_steps=int(args.max_branch_steps),
            rng=rng,
            eos_token_id=tokenizer.eos_token_id,
            show_nested_progress=bool(use_nested),
            progress_position=bar_base + 4,
            progress_desc=f"shard{rank} greedy decode",
            progress_mc_position=bar_base + 2,
            progress_mc_desc=f"shard{rank} greedy MC",
            bucket_group_estimator=None,
        )

        text_minf = tokenizer.decode(response_minf, skip_special_tokens=True)
        text_rand = tokenizer.decode(response_rand, skip_special_tokens=True)
        text_greedy = tokenizer.decode(response_greedy, skip_special_tokens=True)
        ok_minf, eval_minf = evaluate_solution_acc(
            data_source=data_source,
            solution_str=text_minf,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )
        ok_rand, eval_rand = evaluate_solution_acc(
            data_source=data_source,
            solution_str=text_rand,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )
        ok_greedy, eval_greedy = evaluate_solution_acc(
            data_source=data_source,
            solution_str=text_greedy,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )
        ok_minf_boxed, eval_minf_boxed, minf_boxed_answer = _evaluate_final_boxed_only(
            data_source=data_source,
            solution_str=text_minf,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )
        ok_rand_boxed, eval_rand_boxed, rand_boxed_answer = _evaluate_final_boxed_only(
            data_source=data_source,
            solution_str=text_rand,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )
        ok_greedy_boxed, eval_greedy_boxed, greedy_boxed_answer = _evaluate_final_boxed_only(
            data_source=data_source,
            solution_str=text_greedy,
            ground_truth=ground_truth,
            math_eval_backend=str(args.math_eval_backend),
        )
        acc_minf.append(1.0 if ok_minf else 0.0)
        acc_rand.append(1.0 if ok_rand else 0.0)
        acc_greedy.append(1.0 if ok_greedy else 0.0)
        acc_minf_boxed.append(1.0 if ok_minf_boxed else 0.0)
        acc_rand_boxed.append(1.0 if ok_rand_boxed else 0.0)
        acc_greedy_boxed.append(1.0 if ok_greedy_boxed else 0.0)
        branch_count_minf.append(float(len(trace_minf)))
        branch_count_rand.append(float(len(trace_rand)))
        branch_count_greedy.append(float(len(trace_greedy)))
        token_len_minf.append(float(len(response_minf)))
        token_len_rand.append(float(len(response_rand)))
        token_len_greedy.append(float(len(response_greedy)))

        rec: dict[str, Any] = {
            "sample_index": int(global_idx),
            "shard_rank": int(rank),
            "data_source": data_source,
            "ground_truth": ground_truth,
            "prompt_text": prompt_text,
            "entropy_threshold": float(args.entropy_threshold),
            "candidate_top_p": float(args.candidate_top_p),
            "candidate_max_k": int(args.candidate_max_k),
            "max_new_tokens": int(args.max_new_tokens),
            "mc_m_samples": int(args.mc_m_samples),
            "sampling_temperature": float(args.sampling_temperature),
            "sampling_top_p": float(args.sampling_top_p),
            "selection_f_mode": str(args.selection_f_mode),
            "math_eval_backend": str(args.math_eval_backend),
            "force_boxed_answer_instruction": bool(args.force_boxed_answer_instruction),
            "bucket_group_rollouts": int(args.bucket_group_rollouts),
            "bucket_num_bins": int(args.bucket_num_bins),
            "bucket_min_points_per_bin": int(args.bucket_min_points_per_bin),
            "f_continuation_mode": str(args.f_continuation_mode),
            "f_sentence_max_new_tokens": int(args.f_sentence_max_new_tokens),
            "f_sentence_stop": str(args.f_sentence_stop),
            "bias_metrics_mode": str(args.bias_metrics_mode),
            "result_min_f_mc": {
                "is_correct": bool(ok_minf),
                "eval": eval_minf,
                "is_correct_final_boxed_only": bool(ok_minf_boxed),
                "final_boxed_eval": eval_minf_boxed,
                "final_boxed_answer": minf_boxed_answer,
                "response_text": text_minf,
                "response_len_tokens": int(len(response_minf)),
                "num_branch_steps": int(len(trace_minf)),
            },
            "result_random_sampling": {
                "is_correct": bool(ok_rand),
                "eval": eval_rand,
                "is_correct_final_boxed_only": bool(ok_rand_boxed),
                "final_boxed_eval": eval_rand_boxed,
                "final_boxed_answer": rand_boxed_answer,
                "response_text": text_rand,
                "response_len_tokens": int(len(response_rand)),
                "num_branch_steps": int(len(trace_rand)),
            },
            # Backward-compatible alias for older analysis scripts.
            "result_random_topk": {
                "is_correct": bool(ok_rand),
                "eval": eval_rand,
                "is_correct_final_boxed_only": bool(ok_rand_boxed),
                "final_boxed_eval": eval_rand_boxed,
                "final_boxed_answer": rand_boxed_answer,
                "response_text": text_rand,
                "response_len_tokens": int(len(response_rand)),
                "num_branch_steps": int(len(trace_rand)),
            },
            "result_greedy_baseline": {
                "is_correct": bool(ok_greedy),
                "eval": eval_greedy,
                "is_correct_final_boxed_only": bool(ok_greedy_boxed),
                "final_boxed_eval": eval_greedy_boxed,
                "final_boxed_answer": greedy_boxed_answer,
                "response_text": text_greedy,
                "response_len_tokens": int(len(response_greedy)),
                "num_branch_steps": int(len(trace_greedy)),
            },
            "improvement_min_f_over_random": bool(ok_minf and not ok_rand),
            "regression_min_f_over_random": bool((not ok_minf) and ok_rand),
            "improvement_min_f_over_greedy": bool(ok_minf and not ok_greedy),
            "regression_min_f_over_greedy": bool((not ok_minf) and ok_greedy),
        }
        if args.save_traces:
            rec["trace_min_f_mc"] = trace_minf
            rec["trace_random_sampling"] = trace_rand
            # Backward-compatible alias for older analysis scripts.
            rec["trace_random_topk"] = trace_rand
            rec["trace_greedy_baseline"] = trace_greedy

        with open(part_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    file_sync(out_dir=out_dir, rank=rank, world_size=world_size, tag="done_infer_compare")

    if rank == 0:
        merged: list[dict[str, Any]] = []
        for r in range(world_size):
            p = out_dir / f"infer_compare_rank{r}.jsonl"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged.append(json.loads(line))

        merged_path = out_dir / "infer_compare_merged.jsonl"
        with open(merged_path, "w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        paired_improve = sum(int(x.get("improvement_min_f_over_random", False)) for x in merged)
        paired_regress = sum(int(x.get("regression_min_f_over_random", False)) for x in merged)
        paired_improve_g = sum(int(x.get("improvement_min_f_over_greedy", False)) for x in merged)
        paired_regress_g = sum(int(x.get("regression_min_f_over_greedy", False)) for x in merged)
        n = len(merged)
        if n > 0:
            acc_minf_all = [1.0 if x["result_min_f_mc"]["is_correct"] else 0.0 for x in merged]
            acc_minf_boxed_all = [
                1.0 if x["result_min_f_mc"].get("is_correct_final_boxed_only", False) else 0.0 for x in merged
            ]
            acc_rand_all = [
                1.0
                if (x.get("result_random_sampling", x.get("result_random_topk", {})).get("is_correct", False))
                else 0.0
                for x in merged
            ]
            acc_rand_boxed_all = [
                1.0
                if (
                    x.get("result_random_sampling", x.get("result_random_topk", {})).get(
                        "is_correct_final_boxed_only", False
                    )
                )
                else 0.0
                for x in merged
            ]
            acc_greedy_all = [1.0 if x["result_greedy_baseline"]["is_correct"] else 0.0 for x in merged]
            acc_greedy_boxed_all = [
                1.0 if x["result_greedy_baseline"].get("is_correct_final_boxed_only", False) else 0.0 for x in merged
            ]
            branch_minf_all = [float(x["result_min_f_mc"]["num_branch_steps"]) for x in merged]
            branch_rand_all = [
                float(x.get("result_random_sampling", x.get("result_random_topk", {})).get("num_branch_steps", 0.0))
                for x in merged
            ]
            branch_greedy_all = [float(x["result_greedy_baseline"]["num_branch_steps"]) for x in merged]
            len_minf_all = [float(x["result_min_f_mc"]["response_len_tokens"]) for x in merged]
            len_rand_all = [
                float(x.get("result_random_sampling", x.get("result_random_topk", {})).get("response_len_tokens", 0.0))
                for x in merged
            ]
            len_greedy_all = [float(x["result_greedy_baseline"]["response_len_tokens"]) for x in merged]
        else:
            acc_minf_all = acc_minf
            acc_minf_boxed_all = acc_minf_boxed
            acc_rand_all = acc_rand
            acc_rand_boxed_all = acc_rand_boxed
            acc_greedy_all = acc_greedy
            acc_greedy_boxed_all = acc_greedy_boxed
            branch_minf_all = branch_count_minf
            branch_rand_all = branch_count_rand
            branch_greedy_all = branch_count_greedy
            len_minf_all = token_len_minf
            len_rand_all = token_len_rand
            len_greedy_all = token_len_greedy

        summary = {
            "num_prompts": int(n),
            "accuracy_min_f_mc": _mean(acc_minf_all),
            "accuracy_min_f_mc_final_boxed_only": _mean(acc_minf_boxed_all),
            "accuracy_random_sampling": _mean(acc_rand_all),
            "accuracy_random_sampling_final_boxed_only": _mean(acc_rand_boxed_all),
            # Backward-compatible alias.
            "accuracy_random_topk": _mean(acc_rand_all),
            "accuracy_greedy_baseline": _mean(acc_greedy_all),
            "accuracy_greedy_baseline_final_boxed_only": _mean(acc_greedy_boxed_all),
            "accuracy_gain_abs": _mean(acc_minf_all) - _mean(acc_rand_all),
            "accuracy_gain_abs_final_boxed_only": _mean(acc_minf_boxed_all) - _mean(acc_rand_boxed_all),
            "accuracy_gain_abs_vs_greedy": _mean(acc_minf_all) - _mean(acc_greedy_all),
            "accuracy_gain_abs_vs_greedy_final_boxed_only": _mean(acc_minf_boxed_all) - _mean(acc_greedy_boxed_all),
            "paired_improve_count": int(paired_improve),
            "paired_regress_count": int(paired_regress),
            "paired_net_gain": int(paired_improve - paired_regress),
            "paired_improve_count_vs_greedy": int(paired_improve_g),
            "paired_regress_count_vs_greedy": int(paired_regress_g),
            "paired_net_gain_vs_greedy": int(paired_improve_g - paired_regress_g),
            "avg_branch_steps_min_f_mc": _mean(branch_minf_all),
            "avg_branch_steps_random_sampling": _mean(branch_rand_all),
            # Backward-compatible alias.
            "avg_branch_steps_random_topk": _mean(branch_rand_all),
            "avg_branch_steps_greedy_baseline": _mean(branch_greedy_all),
            "avg_response_len_min_f_mc": _mean(len_minf_all),
            "avg_response_len_random_sampling": _mean(len_rand_all),
            # Backward-compatible alias.
            "avg_response_len_random_topk": _mean(len_rand_all),
            "avg_response_len_greedy_baseline": _mean(len_greedy_all),
            "hit_max_len_rate_min_f_mc": _mean(
                [1.0 if float(v) >= float(args.max_new_tokens) else 0.0 for v in len_minf_all]
            ),
            "hit_max_len_rate_random_sampling": _mean(
                [1.0 if float(v) >= float(args.max_new_tokens) else 0.0 for v in len_rand_all]
            ),
            "hit_max_len_rate_greedy_baseline": _mean(
                [1.0 if float(v) >= float(args.max_new_tokens) else 0.0 for v in len_greedy_all]
            ),
            "config": {
                "entropy_threshold": float(args.entropy_threshold),
                "candidate_top_p": float(args.candidate_top_p),
                "candidate_max_k": int(args.candidate_max_k),
                "max_new_tokens": int(args.max_new_tokens),
                "max_branch_steps": int(args.max_branch_steps),
                "mc_m_samples": int(args.mc_m_samples),
                "selection_f_mode": str(args.selection_f_mode),
                "math_eval_backend": str(args.math_eval_backend),
                "bucket_group_rollouts": int(args.bucket_group_rollouts),
                "bucket_num_bins": int(args.bucket_num_bins),
                "bucket_min_points_per_bin": int(args.bucket_min_points_per_bin),
                "mc_temperature": float(args.mc_temperature),
                "mc_top_p": float(args.mc_top_p),
                "sampling_temperature": float(args.sampling_temperature),
                "sampling_top_p": float(args.sampling_top_p),
                "f_continuation_mode": str(args.f_continuation_mode),
                "f_sentence_max_new_tokens": int(args.f_sentence_max_new_tokens),
                "f_sentence_stop": str(args.f_sentence_stop),
                "bias_metrics_mode": str(args.bias_metrics_mode),
            },
        }
        with open(out_dir / "infer_compare_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
