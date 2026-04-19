# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""FEPO: token-level future-entropy advantage shaping (adds sparse bonus to ``advantages``)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import ray
import torch

from verl import DataProto


def select_high_entropy_positions(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    h_threshold: float,
    max_points_per_seq: int,
    max_points_ratio: float,
) -> list[tuple[int, int]]:
    """Return list of (batch_idx, response_time_idx) with entropy >= threshold, top-k per sequence."""
    B, _T = entropy.shape
    out: list[tuple[int, int]] = []
    for b in range(B):
        mask = response_mask[b].bool()
        if not mask.any():
            continue
        row = entropy[b].float().clone()
        row[~mask] = float("-inf")
        valid_idx = torch.where(mask)[0]
        above = valid_idx[row[valid_idx] >= h_threshold]
        if above.numel() == 0:
            continue
        if int(max_points_per_seq) > 0:
            point_budget = int(max_points_per_seq)
        else:
            valid_len = int(mask.sum().item())
            point_budget = max(1, int(valid_len * float(max_points_ratio)))
        k = min(point_budget, int(above.numel()))
        sub = row[above]
        _, topk = torch.topk(sub, k=k, largest=True)
        chosen = above[topk]
        for t in chosen.tolist():
            out.append((b, int(t)))
    return out


def build_fepo_jobs(
    batch: DataProto,
    positions: list[tuple[int, int]],
    *,
    entropy: torch.Tensor,
    f_sentence_max_new_tokens: int,
    mc_max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Branching-aligned jobs: ``prefix_before`` = prompt + response[:t] (excl. token t), ``chosen_token`` = response[t].

    Continuation MC cap matches ``compare_bias_sign_bucket_vs_mc`` style:
    ``min(f_sentence_max_new_tokens, mc_max_new_tokens, response_length - t - 1)``.
    """
    input_ids = batch.batch["input_ids"]
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    pl = batch.batch["prompts"].size(1)

    jobs: list[dict[str, Any]] = []
    coord: list[tuple[int, int]] = []
    for b, t in positions:
        if not response_mask[b, t]:
            continue
        prefix_before = input_ids[b, : pl + t].detach().cpu().tolist()
        chosen_token = int(responses[b, t].item())
        suffix_after: list[int] = []
        for j in range(int(t) + 1, responses.size(1)):
            if not bool(response_mask[b, j]):
                break
            suffix_after.append(int(responses[b, j].item()))
        rem = len(suffix_after)
        if rem <= 0:
            continue
        cont_max = min(int(f_sentence_max_new_tokens), int(mc_max_new_tokens), rem)
        cont_max = max(1, cont_max)
        jobs.append(
            {
                "prefix_before": prefix_before,
                "chosen_token": chosen_token,
                "cont_max_new_tokens": cont_max,
                "h_t": float(entropy[b, t].item()),
                "suffix_after": suffix_after[:cont_max],
            }
        )
        coord.append((b, int(t)))
    return jobs, coord


def deltas_to_sparse_bonus(
    batch_size: int,
    response_length: int,
    device: torch.device,
    dtype: torch.dtype,
    coord: list[tuple[int, int]],
    deltas: list[float],
    ok: list[bool],
    *,
    delta_pos_threshold: float,
    delta_neg_threshold: float,
    bonus_pos: float,
    bonus_neg: float,
) -> torch.Tensor:
    """Map scalar deltas to per-token advantage bonus (sparse)."""
    bonus = torch.zeros(batch_size, response_length, device=device, dtype=dtype)
    for (b, t), d, ok_i in zip(coord, deltas, ok, strict=True):
        if not ok_i or not np.isfinite(d):
            continue
        df = float(d)
        if df >= delta_pos_threshold:
            bonus[b, t] = bonus[b, t] + float(bonus_pos)
        elif df <= -delta_neg_threshold:
            bonus[b, t] = bonus[b, t] - float(bonus_neg)
    return bonus


def run_fepo_advantage_phase(
    batch: DataProto,
    tokenizer: Any,
    fepo_cfg: dict[str, Any],
    async_rollout_manager: Optional[Any],
) -> tuple[DataProto, dict[str, float], list[dict[str, Any]]]:
    """After GRPO advantages are computed, add FEPO token bonus to ``advantages``."""
    if async_rollout_manager is None:
        raise NotImplementedError(
            "FEPO currently requires async rollout (vLLM agent loop). "
            "Set async_rollout_mode / use the standard vLLM agent-loop pipeline."
        )

    enable = bool(fepo_cfg.get("enable", False))
    if not enable:
        return batch, {}, []

    h_threshold = float(fepo_cfg.get("h_threshold", 2.0))
    max_points = int(fepo_cfg.get("max_points_per_seq", 4))
    max_points_ratio = float(fepo_cfg.get("max_points_ratio", 0.01))
    delta_pos_threshold = float(fepo_cfg.get("delta_pos_threshold", 0.05))
    delta_neg_threshold = float(fepo_cfg.get("delta_neg_threshold", 0.05))
    bonus_pos = float(fepo_cfg.get("bonus_pos", 0.02))
    bonus_neg = float(fepo_cfg.get("bonus_neg", 0.02))

    ent_key = "fepo_token_entropy" if "fepo_token_entropy" in batch.batch else "token_entropy"
    if ent_key not in batch.batch:
        raise ValueError(
            "FEPO requires per-token entropy in batch['fepo_token_entropy'] (preferred) or "
            "batch['token_entropy']. Enable recomputation of old_log_prob (non-bypass) or "
            "populate fepo_token_entropy in the trainer."
        )

    entropy = batch.batch[ent_key]
    response_mask = batch.batch["response_mask"]
    positions = select_high_entropy_positions(
        entropy,
        response_mask,
        h_threshold=h_threshold,
        max_points_per_seq=max_points,
        max_points_ratio=max_points_ratio,
    )
    if not positions:
        return batch, {"fepo/n_selected": 0.0}, []

    f_sent = int(fepo_cfg.get("f_sentence_max_new_tokens", 128))
    mc_cap = int(fepo_cfg.get("mc_max_new_tokens", 128))
    jobs, coord = build_fepo_jobs(
        batch,
        positions,
        entropy=entropy,
        f_sentence_max_new_tokens=f_sent,
        mc_max_new_tokens=mc_cap,
    )
    if not jobs:
        return batch, {"fepo/n_selected": float(len(positions)), "fepo/n_jobs_ok": 0.0}, []

    payload = {
        "jobs": jobs,
        "mc_m": int(fepo_cfg.get("mc_m", 1)),
        "mc_temperature": float(fepo_cfg.get("mc_temperature", 1.0)),
        "mc_top_p": float(fepo_cfg.get("mc_top_p", 0.95)),
        "logprobs_k": int(fepo_cfg.get("logprobs_k", 20)),
        "f_continuation_mode": str(fepo_cfg.get("f_continuation_mode", "first_sentence")),
        "f_sentence_max_new_tokens": f_sent,
        "normalize_by_continuation_length": bool(fepo_cfg.get("normalize_by_continuation_length", True)),
        "mc_max_new_tokens": mc_cap,
        "candidate_top_p": float(fepo_cfg.get("candidate_top_p", 0.95)),
        "candidate_max_k": int(fepo_cfg.get("candidate_max_k", 20)),
        "min_candidates": int(fepo_cfg.get("min_candidates", 2)),
        "mc_batch_chunk": int(fepo_cfg.get("mc_batch_chunk", 32)),
        "f_bar_mode": str(fepo_cfg.get("f_bar_mode", "branching")),
        "f_real_mode": str(fepo_cfg.get("f_real_mode", "chosen_branch_mc")),
    }

    async_rollout_manager.wake_up()
    try:
        out = ray.get(async_rollout_manager.server_handles[0].fepo_compute.remote(payload))
    finally:
        async_rollout_manager.sleep()

    deltas = out.get("deltas", [])
    ok = out.get("ok", [])
    details = out.get("details", [])
    advantages = batch.batch["advantages"]
    bonus = deltas_to_sparse_bonus(
        advantages.size(0),
        advantages.size(1),
        advantages.device,
        advantages.dtype,
        coord,
        deltas,
        ok,
        delta_pos_threshold=delta_pos_threshold,
        delta_neg_threshold=delta_neg_threshold,
        bonus_pos=bonus_pos,
        bonus_neg=bonus_neg,
    )
    batch.batch["advantages"] = advantages + bonus * response_mask.float()

    n_ok = float(sum(1 for x in ok if x))
    min_f_vals = [
        float(d.get("branch_min_f_mc"))
        for d, k in zip(details, ok, strict=False)
        if k and isinstance(d, dict) and d.get("branch_min_f_mc") is not None
    ]
    metrics = {
        "fepo/n_selected": float(len(positions)),
        "fepo/n_jobs_ok": n_ok,
        "fepo/mean_delta": float(np.nanmean([float(d) for d, k in zip(deltas, ok, strict=True) if k]))
        if any(ok)
        else 0.0,
        "fepo/branch_steps_min_f_mc": float(np.nanmean(min_f_vals)) if min_f_vals else 0.0,
    }
    point_records: list[dict[str, Any]] = []
    for i, ((b, t), job) in enumerate(zip(coord, jobs, strict=False)):
        d = details[i] if i < len(details) and isinstance(details[i], dict) else {}
        rec: dict[str, Any] = {
            "batch_index": int(b),
            "response_t": int(t),
            "ok": bool(ok[i]) if i < len(ok) else False,
            "delta": float(deltas[i]) if i < len(deltas) else None,
            "h_t": float(job.get("h_t", 0.0)),
            "prefix_before_token_ids": list(job.get("prefix_before", [])),
            "chosen_token_id": int(job.get("chosen_token", -1)),
            "suffix_after_token_ids": list(job.get("suffix_after", [])),
            "f_bar": d.get("f_bar"),
            "f_real": d.get("f_real"),
            "branch_min_f_mc": d.get("branch_min_f_mc"),
            "f_bar_mode": d.get("f_bar_mode"),
            "f_real_mode": d.get("f_real_mode"),
            "cands": d.get("cands"),
            "cand_probs": d.get("cand_probs"),
            "f_mc": d.get("f_mc"),
            "reason": d.get("reason"),
        }
        # Keep readable context for offline inspection.
        try:
            rec["prefix_before_text"] = tokenizer.decode(rec["prefix_before_token_ids"], skip_special_tokens=True)
            rec["chosen_token_text"] = tokenizer.decode([rec["chosen_token_id"]], skip_special_tokens=True)
            rec["suffix_after_text"] = tokenizer.decode(rec["suffix_after_token_ids"], skip_special_tokens=True)
            rec["context_text"] = rec["prefix_before_text"] + rec["chosen_token_text"]
        except Exception:
            pass
        point_records.append(rec)
    return batch, metrics, point_records
