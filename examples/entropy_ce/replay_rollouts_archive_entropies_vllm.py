#!/usr/bin/env python3
"""为 rollouts_archive*.jsonl 补全逐步 token 熵（无历史 entropies 时）。

输入每行 JSON 需含 ``prompt_text`` 与 ``rollouts``；每个 rollout 至少含 ``response_text``。
对每条 rollout：用 tokenizer 将 ``response_text`` 编成 token，再对 t=0..n-1 用 vLLM 在
``prompt_ids + response_ids[:t]`` 上取下一步 top-k logprobs，计算与实验一致的 top-k 重归一化熵。

输出与输入结构相同，但每个 rollout 增加 ``entropies``（长度与 response token 数一致）。

多卡：传 ``--vllm_shard_rank`` / ``--vllm_shard_world_size``，按行号分片；rank0 合并为
``rollouts_archive_with_entropies_merged.jsonl``。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from entropy_credit_experiment import (
    _configure_vllm_ipc_for_single_node,
    _configure_vllm_multiprocessing_spawn,
    _restore_torchrun_dist_env,
    _snapshot_and_clear_torchrun_dist_env,
    entropy_from_logprobs_topk,
    file_sync,
    init_dist,
    purge_all_torchrun_like_env_for_vllm_standalone,
)
from infer_topk_f_mc_compare import _step_logprobs_vllm

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _entropies_for_response(
    llm: Any,
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    logprobs_k: int,
) -> list[float]:
    out: list[float] = []
    for t in range(len(response_ids)):
        prefix = prompt_ids + response_ids[:t]
        _, step_lp = _step_logprobs_vllm(llm=llm, prefix_ids=prefix, logprobs_k=logprobs_k)
        out.append(float(entropy_from_logprobs_topk(step_lp)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay per-step entropies for rollouts_archive jsonl.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vllm_logprobs_topk", type=int, default=20)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--vllm_shard_rank", type=int, default=None)
    parser.add_argument("--vllm_shard_world_size", type=int, default=None)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

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

    in_path = Path(args.input_jsonl).expanduser().resolve()
    if not in_path.is_file():
        raise SystemExit(f"input not found: {in_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"rollouts_with_entropies_rank{rank}.jsonl"
    part_path.write_text("", encoding="utf-8")

    lines: list[str] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)

    local_indices = [i for i in range(len(lines)) if i % world_size == rank]
    use_tqdm = tqdm is not None and not args.no_progress and rank == 0
    it = local_indices
    if use_tqdm:
        it = tqdm(it, total=len(local_indices), desc=f"shard{rank} replay")

    n_ok = 0
    n_skip = 0
    for idx in it:
        rec = json.loads(lines[idx])
        if not isinstance(rec, dict):
            n_skip += 1
            continue
        prompt_text = rec.get("prompt_text")
        rollouts = rec.get("rollouts")
        if not isinstance(prompt_text, str) or not isinstance(rollouts, list):
            n_skip += 1
            continue

        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        new_rollouts: list[dict[str, Any]] = []
        for r in rollouts:
            if not isinstance(r, dict):
                continue
            rt = r.get("response_text")
            if not isinstance(rt, str) or not rt.strip():
                new_rollouts.append(dict(r))
                continue
            resp_ids = tokenizer(rt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
            if not resp_ids:
                new_rollouts.append({**r, "entropies": []})
                continue
            try:
                ent = _entropies_for_response(
                    llm,
                    prompt_ids=prompt_ids,
                    response_ids=resp_ids,
                    logprobs_k=int(args.vllm_logprobs_topk),
                )
            except Exception:
                new_rollouts.append(dict(r))
                n_skip += 1
                continue
            rr = dict(r)
            rr["entropies"] = ent
            rr["response_length_tokens_replayed"] = len(resp_ids)
            new_rollouts.append(rr)

        out_rec = dict(rec)
        out_rec["rollouts"] = new_rollouts
        out_rec["entropy_replay_note"] = (
            "entropies from vLLM top-k step entropy on replay tokenization of response_text; "
            "align with training if tokenizer round-trip matches."
        )
        with open(part_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        n_ok += 1

    file_sync(out_dir=out_dir, rank=rank, world_size=world_size, tag="done_replay_entropy")

    if rank == 0:
        merged: list[dict[str, Any]] = []
        for r in range(world_size):
            p = out_dir / f"rollouts_with_entropies_rank{r}.jsonl"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged.append(json.loads(line))
        merged.sort(key=lambda x: int(x.get("sample_index", 0)))
        merged_path = out_dir / "rollouts_archive_with_entropies_merged.jsonl"
        with open(merged_path, "w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta = {
            "input_jsonl": str(in_path),
            "output_merged": str(merged_path),
            "n_lines_input": len(lines),
            "n_lines_output": len(merged),
            "n_ok_shard0_report": n_ok,
            "n_skip_shard0_report": n_skip,
        }
        with open(out_dir / "replay_entropy_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
