#!/usr/bin/env python3
"""
Delta-H entropy credit experiment for VERL-format math data.

Features:
- Multi-GPU analysis via torchrun.
- Single model path input.
- VERL-format parquet/json input support.
- max_samples support.
- Phase2 defaults to Method B (outcome flip rate).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_dapo as math_dapo_score

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def barrier() -> None:
    if is_dist():
        dist.barrier()


def file_sync(out_dir: Path, rank: int, world_size: int, tag: str = "done", poll_s: float = 2.0) -> None:
    """File-based sync to avoid NCCL/TCPStore barrier timeouts."""
    marker = out_dir / f"_{tag}_rank{rank}"
    marker.write_text("ok\n", encoding="utf-8")
    if rank != 0:
        return
    expected = [out_dir / f"_{tag}_rank{r}" for r in range(world_size)]
    while True:
        if all(p.exists() for p in expected):
            return
        time.sleep(poll_s)


def configure_cuda_visible_one_gpu_per_rank_for_vllm() -> None:
    """vLLM expects a single visible GPU per engine; map physical GPU via LOCAL_RANK."""
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(lr)


# torchrun/torchelastic injects these; vLLM EngineCore subprocesses inherit them and may try to
# join the *same* TCPStore as this script, causing 600s timeouts. Strip while constructing LLM().
_TORCHRUN_DIST_ENV_KEYS = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
)


def _snapshot_and_clear_torchrun_dist_env() -> dict[str, str | None]:
    snap: dict[str, str | None] = {}
    for k in _TORCHRUN_DIST_ENV_KEYS:
        snap[k] = os.environ.pop(k, None)
    return snap


def _restore_torchrun_dist_env(snap: dict[str, str | None]) -> None:
    for k, v in snap.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _configure_vllm_multiprocessing_spawn() -> None:
    """vLLM v1 EngineCore may fork workers; if PyTorch CUDA was touched in parent, fork breaks.

    Force spawn before constructing LLM(). Safe to call once per torchrun worker process.
    """
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set (e.g. by another library); ignore.
        pass
    # Some vLLM builds read this; harmless if ignored.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def init_dist(backend: str = "hf") -> tuple[int, int, int]:
    """When backend=vllm, each process only sees one GPU (cuda:0 == physical LOCAL_RANK)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "vllm":
        local_rank = 0
    else:
        local_rank = local_rank_env
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def load_data(path: str, max_samples: int, seed: int) -> list[dict[str, Any]]:
    if path.endswith(".parquet"):
        ds = datasets.load_dataset("parquet", data_files=path)["train"]
    elif path.endswith(".json") or path.endswith(".jsonl"):
        ds = datasets.load_dataset("json", data_files=path)["train"]
    else:
        raise ValueError(f"Unsupported input format: {path}")

    total = len(ds)
    if 0 < max_samples < total:
        indices = list(range(total))
        random.Random(seed).shuffle(indices)
        ds = ds.select(indices[:max_samples])
    return [ds[i] for i in range(len(ds))]


def build_prompt_text(tokenizer, prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, list):
        return tokenizer.apply_chat_template(prompt_obj, add_generation_prompt=True, tokenize=False)
    raise ValueError(f"Unsupported prompt type: {type(prompt_obj)}")


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-20))
    return float((-probs * log_probs).sum().item())


def entropy_from_logprobs_topk(logprobs_dict: dict[int, float]) -> float:
    """Entropy over renormalized top-K logprobs (vLLM); not identical to full-vocab softmax."""
    if not logprobs_dict:
        return 0.0
    lps = np.array(list(logprobs_dict.values()), dtype=np.float64)
    m = float(np.max(lps))
    p = np.exp(lps - m)
    z = float(p.sum())
    if z <= 0.0 or not np.isfinite(z):
        return 0.0
    p = p / z
    return float(-np.sum(p * np.log(np.clip(p, 1e-20, 1.0))))


def varentropy_from_logprobs_topk(logprobs_dict: dict[int, float], entropy: float) -> float:
    """Match HF varentropy form on renormalized top-K: sum q_i (log q_i + H)^2."""
    if not logprobs_dict:
        return 0.0
    lps = np.array(list(logprobs_dict.values()), dtype=np.float64)
    m = float(np.max(lps))
    p = np.exp(lps - m)
    z = float(p.sum())
    if z <= 0.0 or not np.isfinite(z):
        return 0.0
    p = p / z
    log_q = (lps - m) - math.log(z + 1e-20)
    center = log_q + entropy
    return float(np.sum(p * (center**2)))


def suffix_avg(values: list[float], start: int) -> float:
    if start >= len(values):
        return 0.0
    tail = values[start:]
    return float(sum(tail) / len(tail))


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)
    xr = rankdata_average_ties(xa)
    yr = rankdata_average_ties(ya)
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    denom = math.sqrt(float((xc * xc).sum() * (yc * yc).sum()))
    if denom == 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def precision_at_k(signal: list[float], importance: list[float], k: int) -> float:
    if not signal or not importance:
        return float("nan")
    k = min(k, len(signal), len(importance))
    if k <= 0:
        return float("nan")
    sig_idx = np.argsort(np.array(signal))[-k:]
    imp_idx = set(np.argsort(np.array(importance))[-k:].tolist())
    hit = sum(int(i in imp_idx) for i in sig_idx.tolist())
    return float(hit / k)


def extract_acc(result: Any) -> bool:
    """Normalize reward output to a boolean correctness flag."""
    if isinstance(result, dict) and "acc" in result:
        return bool(result["acc"])
    return float(result) > 0.5


def evaluate_solution_acc(data_source: str, solution_str: str, ground_truth: str) -> tuple[bool, dict[str, Any]]:
    """Evaluate correctness with math_dapo boxed-answer logic for math-like sources."""
    if data_source in {"math_dapo", "math", "math_dapo_reasoning"} or data_source.startswith("aime"):
        res = math_dapo_score.compute_score(solution_str, ground_truth, strict_box_verify=True)
        return bool(res.get("acc", False)), {"mode": "math_dapo_strict_box", **res}

    res = default_compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth)
    return extract_acc(res), {"mode": "default_compute_score", "raw_result": res}


def build_token_context(tokenizer, response_ids: list[int], pos: int, window: int) -> dict[str, Any]:
    left = max(0, pos - window)
    right = min(len(response_ids), pos + window + 1)
    return {
        "branch_token_index": pos,
        "context_window": window,
        "context_left_text": tokenizer.decode(response_ids[left:pos], skip_special_tokens=True),
        "branch_token_text": tokenizer.decode([response_ids[pos]], skip_special_tokens=True),
        "context_right_text": tokenizer.decode(response_ids[pos + 1 : right], skip_special_tokens=True),
    }


@dataclass
class RolloutItem:
    sample_index: int
    rollout_index: int
    data_source: str
    ground_truth: str
    prompt_text: str
    response_text: str
    response_token_ids: list[int]
    entropies: list[float]
    deltas: list[float]
    varentropies: list[float]
    branching_factor: list[float]
    importance_method_b: list[float]
    score_backend: str = "hf"
    entropy_note: str = "full_vocab_softmax"


def compute_varentropy(logits: torch.Tensor, entropy: float) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-20))
    center = log_probs + entropy
    return float((probs * (center * center)).sum().item())


def generate_rollout(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[int], list[torch.Tensor]]:
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    attn = encoded["attention_mask"].to(model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out.sequences[0, input_ids.shape[1] :].tolist()
    scores = [s[0].detach().float().cpu() for s in out.scores]
    return gen_ids, scores


def vllm_generate_quiet(llm: Any, prompts: list, sampling_params: Any) -> Any:
    """Call vLLM without its per-request tqdm (conflicts with our outer tqdm)."""
    sig = inspect.signature(llm.generate)
    if "use_tqdm" in sig.parameters:
        return llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    return llm.generate(prompts, sampling_params=sampling_params)


def generate_rollout_vllm(
    llm: Any,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logprobs_k: int,
) -> tuple[list[int], list[dict[int, float]]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].tolist()
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs_k,
    )
    outputs = vllm_generate_quiet(llm, [TokensPrompt(prompt_token_ids=prompt_ids)], sp)
    o = outputs[0].outputs[0]
    gen_ids = list(o.token_ids)
    logprobs_per_step: list[dict[int, float]] = []
    for step_lp in o.logprobs or []:
        d: dict[int, float] = {}
        for tid, info in step_lp.items():
            d[int(tid)] = float(info.logprob)
        logprobs_per_step.append(d)
    while len(logprobs_per_step) < len(gen_ids):
        logprobs_per_step.append({})
    return gen_ids, logprobs_per_step


def method_b_importance(
    model: Any,
    tokenizer,
    prompt_text: str,
    response_ids: list[int],
    scores: list[torch.Tensor] | list[dict[int, float]],
    data_source: str,
    ground_truth: str,
    selected_positions: list[int],
    m_samples: int,
    top_k_alt: int,
    show_progress: bool = False,
    progress_position: int = 2,
    progress_desc: str = "phase2 candidates",
    context_window_tokens: int = 24,
    backend: str = "hf",
    llm: Any | None = None,
) -> tuple[list[float], list[dict[str, Any]], bool]:
    base_response = tokenizer.decode(response_ids, skip_special_tokens=True)
    base_acc, base_eval = evaluate_solution_acc(data_source=data_source, solution_str=base_response, ground_truth=ground_truth)

    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].tolist()
    importance = [0.0 for _ in range(len(response_ids))]
    trace_records: list[dict[str, Any]] = []

    pos_iter = selected_positions
    if show_progress and tqdm is not None:
        pos_iter = tqdm(
            selected_positions,
            total=len(selected_positions),
            desc=progress_desc,
            dynamic_ncols=True,
            position=progress_position,
            leave=False,
        )

    for pos in pos_iter:
        if pos < 0 or pos >= len(response_ids) or pos >= len(scores):
            continue
        step_sc = scores[pos]
        if isinstance(step_sc, dict):
            lp = step_sc
            if not lp:
                continue
            tk = min(max(2, top_k_alt), len(lp))
            items = sorted(lp.items(), key=lambda x: x[1], reverse=True)[:tk]
            candidates = [tid for tid, _ in items]
            vals = np.array([v for _, v in items], dtype=np.float64)
            m = float(np.max(vals))
            expv = np.exp(vals - m)
            zsum = float(expv.sum())
            if zsum <= 0.0 or not np.isfinite(zsum):
                continue
            idx_map = {tid: i for i, (tid, _) in enumerate(items)}
            alt_tokens = [c for c in candidates if c != response_ids[pos]]
            if not alt_tokens:
                continue
            alt_probs = np.array([expv[idx_map[t]] for t in alt_tokens], dtype=np.float64)
            alt_probs = alt_probs / alt_probs.sum()
        else:
            logits = step_sc
            tk = min(max(2, top_k_alt), logits.shape[-1])
            topk = torch.topk(logits, k=tk, dim=-1)
            candidates = topk.indices.tolist()
            topk_values = topk.values.detach().float().cpu().numpy()
            alt_tokens = [c for c in candidates if c != response_ids[pos]]
            if not alt_tokens:
                continue
            alt_scores = np.array([topk_values[candidates.index(t)] for t in alt_tokens], dtype=np.float64)
            finite_mask = np.isfinite(alt_scores)
            if not np.any(finite_mask):
                alt_probs = np.ones(len(alt_tokens), dtype=np.float64) / float(len(alt_tokens))
            else:
                safe_scores = alt_scores[finite_mask]
                safe_scores = safe_scores - np.max(safe_scores)
                safe_probs = np.exp(safe_scores)
                safe_sum = float(safe_probs.sum())
                if (not np.isfinite(safe_sum)) or safe_sum <= 0.0:
                    alt_probs = np.ones(len(alt_tokens), dtype=np.float64) / float(len(alt_tokens))
                else:
                    alt_probs = np.zeros(len(alt_tokens), dtype=np.float64)
                    alt_probs[finite_mask] = safe_probs / safe_sum
                    total = float(alt_probs.sum())
                    if (not np.isfinite(total)) or total <= 0.0:
                        alt_probs = np.ones(len(alt_tokens), dtype=np.float64) / float(len(alt_tokens))
                    else:
                        alt_probs = alt_probs / total

        flips = 0
        pos_ctx = build_token_context(tokenizer, response_ids, pos, context_window_tokens)
        for sample_id in range(m_samples):
            sampled = int(np.random.choice(np.array(alt_tokens), p=alt_probs))
            mutated = response_ids[:]
            mutated[pos] = sampled

            prefix = prompt_ids + mutated[: pos + 1]
            rem_len = max(1, len(response_ids) - pos - 1)
            if backend == "vllm":
                from vllm import SamplingParams
                from vllm.inputs import TokensPrompt

                assert llm is not None
                sp = SamplingParams(max_tokens=rem_len, temperature=1.0, top_p=0.95)
                out = vllm_generate_quiet(llm, [TokensPrompt(prompt_token_ids=prefix)], sp)
                new_ids = list(out[0].outputs[0].token_ids)
            else:
                prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=model.device).unsqueeze(0)
                attn = torch.ones_like(prefix_tensor)
                out = model.generate(
                    input_ids=prefix_tensor,
                    attention_mask=attn,
                    max_new_tokens=rem_len,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                new_ids = out.sequences[0, prefix_tensor.shape[1] :].tolist()
            full_mutated = mutated[: pos + 1] + new_ids
            new_resp = tokenizer.decode(full_mutated, skip_special_tokens=True)
            acc, cf_eval = evaluate_solution_acc(data_source=data_source, solution_str=new_resp, ground_truth=ground_truth)
            is_flip = acc != base_acc
            if is_flip:
                flips += 1
            trace_records.append(
                {
                    "branch_position": pos_ctx,
                    "sample_id": sample_id,
                    "original_token_id": int(response_ids[pos]),
                    "original_token_text": tokenizer.decode([response_ids[pos]], skip_special_tokens=True),
                    "replaced_token_id": int(sampled),
                    "replaced_token_text": tokenizer.decode([sampled], skip_special_tokens=True),
                    "candidate_token_ids": [int(t) for t in alt_tokens],
                    "candidate_probs": [float(p) for p in alt_probs.tolist()],
                    "mutated_prefix_text": tokenizer.decode(mutated[: pos + 1], skip_special_tokens=True),
                    "generated_suffix_text": tokenizer.decode(new_ids, skip_special_tokens=True),
                    "counterfactual_response_text": new_resp,
                    "base_acc": bool(base_acc),
                    "counterfactual_acc": bool(acc),
                    "flip": bool(is_flip),
                    "base_eval": base_eval,
                    "counterfactual_eval": cf_eval,
                }
            )
        importance[pos] = float(flips / m_samples)
    return importance, trace_records, base_acc


def pick_candidate_positions(deltas: list[float], entropies: list[float], seed: int) -> list[int]:
    n = len(entropies)
    if n == 0:
        return []
    k = max(1, int(math.ceil(n * 0.1)))
    idx_delta = np.argsort(np.array(deltas))[-k:].tolist()
    idx_entropy = np.argsort(np.array(entropies))[-k:].tolist()
    rng = random.Random(seed)
    all_idx = list(range(n))
    rng.shuffle(all_idx)
    idx_random = all_idx[:k]
    return sorted(set(idx_delta + idx_entropy + idx_random))


def cap_positions(positions: list[int], cap: int, seed: int) -> list[int]:
    """Cap candidate positions to avoid Method-B blowup.

    Keep a deterministic random subset so the mix (delta/high-H/random) is roughly preserved.
    """
    if cap <= 0 or len(positions) <= cap:
        return positions
    rng = random.Random(seed)
    pos = positions[:]
    rng.shuffle(pos)
    return sorted(pos[:cap])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="VERL format parquet/json/jsonl.")
    parser.add_argument("--model_path", type=str, required=True, help="Single model path.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--rollouts_per_prompt", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--phase2_method", type=str, default="B", choices=["A", "B"])
    parser.add_argument("--method_b_m_samples", type=int, default=4)
    parser.add_argument("--method_b_topk_alt", type=int, default=10)
    parser.add_argument(
        "--phase2_max_positions",
        type=int,
        default=64,
        help="Cap the number of candidate token positions for Phase2 (Method B). Set <=0 to disable cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--progress_all_ranks",
        action="store_true",
        help="Show a progress bar on every rank (default: rank 0 only).",
    )
    parser.add_argument(
        "--phase2_progress",
        action="store_true",
        help="Show tqdm progress inside Phase2 (Method B) candidate loop (rank 0 by default).",
    )
    parser.add_argument(
        "--save_case_traces",
        action="store_true",
        help="Save per-case branch traces to output_dir/cases/case_<sample_index>.jsonl.",
    )
    parser.add_argument(
        "--context_window_tokens",
        type=int,
        default=24,
        help="Number of response tokens to keep on each side of the branch position.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "vllm"],
        help="hf: HuggingFace generate + full-vocab logits. vllm: faster inference; entropies from top-K logprobs.",
    )
    parser.add_argument(
        "--vllm_logprobs_topk",
        type=int,
        default=256,
        help="When backend=vllm: number of top logprobs per generated token (for Delta/H/V).",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="When backend=vllm: vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=32768,
        help="When backend=vllm: max_model_len for vLLM engine (prompt + max_new_tokens).",
    )
    args = parser.parse_args()

    if args.backend == "vllm":
        # Must run before LLM(): v1 EngineCore subprocess + fork breaks if CUDA was initialized in parent.
        _configure_vllm_multiprocessing_spawn()
        configure_cuda_visible_one_gpu_per_rank_for_vllm()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = None
    llm: Any = None
    if args.backend == "hf":
        local_rank, rank, world_size = init_dist(backend="hf")
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map={"": local_rank} if torch.cuda.is_available() else "cpu",
        )
        model.eval()
    else:
        # Avoid vLLM EngineCore inheriting torchrun's MASTER_ADDR/PORT and joining the wrong store.
        dist_snap = _snapshot_and_clear_torchrun_dist_env()
        try:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
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
        local_rank, rank, world_size = init_dist(backend="vllm")

    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out_dir / "cases"
    if args.save_case_traces:
        case_dir.mkdir(parents=True, exist_ok=True)
    part_file = out_dir / f"phase1_3_rank{rank}.jsonl"
    with open(part_file, "w", encoding="utf-8"):
        pass

    rows = load_data(args.input_data, args.max_samples, args.seed)
    local_rows = [row for idx, row in enumerate(rows) if idx % world_size == rank]

    env_no_tqdm = os.environ.get("TQDM_DISABLE", "").strip().lower() in ("1", "true", "yes")
    use_tqdm = tqdm is not None and not args.no_progress and not env_no_tqdm
    pbar_ranks = range(world_size) if (use_tqdm and args.progress_all_ranks) else [0]
    show_outer = use_tqdm and rank in pbar_ranks
    show_inner = use_tqdm and rank in pbar_ranks

    outer_it = enumerate(local_rows)
    if show_outer:
        bar_pos = rank * 2 if args.progress_all_ranks else 0
        outer_it = tqdm(
            outer_it,
            total=len(local_rows),
            desc=f"rank{rank} samples",
            dynamic_ncols=True,
            position=bar_pos,
            leave=True,
        )

    for local_i, row in outer_it:
        global_idx = local_i * world_size + rank
        prompt_text = build_prompt_text(tokenizer, row["prompt"])
        data_source = row.get("data_source", "math_dapo")
        ground_truth = str((row.get("reward_model") or {}).get("ground_truth", ""))

        rollout_it = range(args.rollouts_per_prompt)
        if show_inner:
            inner_pos = rank * 2 + 1 if args.progress_all_ranks else 1
            rollout_it = tqdm(
                rollout_it,
                total=args.rollouts_per_prompt,
                desc=f"rank{rank} rollouts",
                dynamic_ncols=True,
                position=inner_pos,
                leave=False,
            )
        for rollout_idx in rollout_it:
            if args.backend == "vllm":
                assert llm is not None
                response_ids, scores = generate_rollout_vllm(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    logprobs_k=args.vllm_logprobs_topk,
                )
            else:
                assert model is not None
                response_ids, scores = generate_rollout(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            if not response_ids:
                continue
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            base_acc, base_eval = evaluate_solution_acc(
                data_source=data_source, solution_str=response_text, ground_truth=ground_truth
            )
            if args.backend == "vllm":
                entropies = [entropy_from_logprobs_topk(s) for s in scores[: len(response_ids)]]
                varentropies = [
                    varentropy_from_logprobs_topk(scores[i], entropies[i]) for i in range(len(entropies))
                ]
            else:
                entropies = [entropy_from_logits(s) for s in scores[: len(response_ids)]]
                varentropies = [compute_varentropy(s, h) for s, h in zip(scores[: len(response_ids)], entropies, strict=False)]
            branching = [math.exp(h) for h in entropies]
            deltas = []
            for t in range(len(entropies)):
                et = suffix_avg(entropies, t + 1)
                et1 = suffix_avg(entropies, t + 2)
                deltas.append(et - et1)

            if args.phase2_method == "B":
                cand_seed = args.seed + global_idx + rollout_idx
                candidates = pick_candidate_positions(deltas, entropies, cand_seed)
                candidates = cap_positions(candidates, args.phase2_max_positions, cand_seed)
                importance, branch_traces, base_acc_from_method_b = method_b_importance(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    response_ids=response_ids,
                    scores=scores,
                    data_source=data_source,
                    ground_truth=ground_truth,
                    selected_positions=candidates,
                    m_samples=args.method_b_m_samples,
                    top_k_alt=args.method_b_topk_alt,
                    show_progress=bool(args.phase2_progress and show_inner),
                    progress_position=(rank * 2 + 2) if args.progress_all_ranks else 2,
                    progress_desc=f"rank{rank} phase2",
                    context_window_tokens=args.context_window_tokens,
                    backend=args.backend,
                    llm=llm,
                )
                base_acc = bool(base_acc_from_method_b)
            else:
                importance = [0.0 for _ in range(len(response_ids))]
                branch_traces = []
                candidates = []

            item = RolloutItem(
                sample_index=global_idx,
                rollout_index=rollout_idx,
                data_source=data_source,
                ground_truth=ground_truth,
                prompt_text=prompt_text,
                response_text=response_text,
                response_token_ids=response_ids,
                entropies=entropies,
                deltas=deltas,
                varentropies=varentropies,
                branching_factor=branching,
                importance_method_b=importance,
                score_backend=args.backend,
                entropy_note=(
                    f"vllm_top{args.vllm_logprobs_topk}_renorm"
                    if args.backend == "vllm"
                    else "full_vocab_softmax"
                ),
            )
            with open(part_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
            if args.save_case_traces:
                case_path = case_dir / f"case_{global_idx}.jsonl"
                with open(case_path, "a", encoding="utf-8") as cf:
                    cf.write(
                        json.dumps(
                            {
                                "event": "rollout_base",
                                "rank": rank,
                                "sample_index": global_idx,
                                "rollout_index": rollout_idx,
                                "data_source": data_source,
                                "ground_truth": ground_truth,
                                "prompt_text": prompt_text,
                                "response_text": response_text,
                                "response_token_ids": response_ids,
                                "base_acc": bool(base_acc),
                                "base_eval": base_eval,
                                "phase2_candidates": candidates if args.phase2_method == "B" else [],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    for tr in branch_traces:
                        tr_line = {
                            "event": "branch_counterfactual",
                            "rank": rank,
                            "sample_index": global_idx,
                            "rollout_index": rollout_idx,
                            "data_source": data_source,
                            "ground_truth": ground_truth,
                            "prompt_text": prompt_text,
                            **tr,
                        }
                        cf.write(json.dumps(tr_line, ensure_ascii=False) + "\n")

    # Avoid NCCL barrier timeout when ranks finish at very different times.
    file_sync(out_dir=out_dir, rank=rank, world_size=world_size, tag="done")

    if rank == 0:
        all_records = []
        for r in range(world_size):
            fpath = out_dir / f"phase1_3_rank{r}.jsonl"
            if not fpath.exists():
                continue
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))
        merged_path = out_dir / "phase1_3_merged.jsonl"
        with open(merged_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        corr_delta, corr_h, corr_v = [], [], []
        p5_delta, p10_delta, p20_delta = [], [], []
        p5_h, p10_h, p20_h = [], [], []
        for rec in all_records:
            imp = rec["importance_method_b"]
            if not any(x > 0 for x in imp):
                continue
            delta = rec["deltas"]
            h = rec["entropies"]
            v = rec["varentropies"]
            corr_delta.append(spearman(delta, imp))
            corr_h.append(spearman(h, imp))
            corr_v.append(spearman(v, imp))
            p5_delta.append(precision_at_k(delta, imp, 5))
            p10_delta.append(precision_at_k(delta, imp, 10))
            p20_delta.append(precision_at_k(delta, imp, 20))
            p5_h.append(precision_at_k(h, imp, 5))
            p10_h.append(precision_at_k(h, imp, 10))
            p20_h.append(precision_at_k(h, imp, 20))

        def mean_no_nan(x: list[float]) -> float:
            arr = np.array(x, dtype=np.float64)
            if arr.size == 0:
                return float("nan")
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return float("nan")
            return float(arr.mean())

        metrics = {
            "num_rollouts": len(all_records),
            "phase2_method": args.phase2_method,
            "spearman_delta_vs_importance": mean_no_nan(corr_delta),
            "spearman_entropy_vs_importance": mean_no_nan(corr_h),
            "spearman_varentropy_vs_importance": mean_no_nan(corr_v),
            "precision@5_delta": mean_no_nan(p5_delta),
            "precision@10_delta": mean_no_nan(p10_delta),
            "precision@20_delta": mean_no_nan(p20_delta),
            "precision@5_entropy": mean_no_nan(p5_h),
            "precision@10_entropy": mean_no_nan(p10_h),
            "precision@20_entropy": mean_no_nan(p20_h),
        }
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
