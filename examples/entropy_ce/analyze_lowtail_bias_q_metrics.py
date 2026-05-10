#!/usr/bin/env python3
"""Aggregate bias>0 accuracy and recall from low-tail bias validation JSONL.

Reads records produced by collect_lowtail_bias_validation.py (fields: q, bias,
f_suffix_rate, suffix_tokens_used, etc.).

Definitions (aligned with collect script summary):
- q: rank percentile of length-bucket-normalized suffix entropy rate; low-tail
  uses small q (q <= cutoff).
- Accuracy: P(bias > 0 | q <= cutoff) among tokens with finite q in the tail set.
- Recall: among tokens with bias > 0 and |bias| > tau, fraction with q <= cutoff.

Default behavior (``--recompute-q``, on by default):
  Load **all** given JSONL files into one list, then on that **full union**
  recompute length-bucket de-biasing for ``f_suffix_rate`` →
  ``f_suffix_rate_norm``, then assign global rank-percentile ``q``. This
  matches rank-0's ``lowtail_bias_points.jsonl`` pipeline and is **not** the
  per-rank ``q`` stored inside each ``*_rank*.jsonl`` shard.

Use ``--no-recompute-q`` only if you intentionally trust ``q`` already in the
file (e.g. a single merged JSONL). With multiple shards, ``--no-recompute-q``
is usually wrong.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


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


def _load_jsonl_paths(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _expand_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in inputs:
        if not s:
            continue
        p = Path(s).expanduser()
        if any(ch in str(p) for ch in "*?["):
            matched = sorted(glob_module.glob(str(p)))
            out.extend(Path(x) for x in matched)
        else:
            out.append(p.resolve())
    # Stable de-dupe by resolved path
    seen: set[Path] = set()
    deduped: list[Path] = []
    for x in out:
        r = x.resolve()
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def _paths_from_dir(dir_path: Path, name_glob: str) -> list[Path]:
    if not dir_path.is_dir():
        raise SystemExit(f"Not a directory: {dir_path}")
    return sorted(dir_path.glob(name_glob))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        nargs="*",
        default=[],
        help="One or more JSONL paths; shell globs may not expand on all setups — prefer --from-dir.",
    )
    parser.add_argument(
        "--from-dir",
        type=str,
        default="",
        help="Directory to scan with --glob-name (avoids empty shell glob / wrong path).",
    )
    parser.add_argument(
        "--glob-name",
        type=str,
        default="lowtail_bias_points_rank*.jsonl",
        help="Pattern for Path.glob under --from-dir. Try low_bias_points_rank*.jsonl if needed.",
    )
    parser.add_argument(
        "--bias-abs-threshold",
        type=float,
        default=0.0,
        help="Recall denominator/numerator only counts bias>0 with |bias| > this (default 0).",
    )
    parser.add_argument(
        "--q-cutoffs",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.5],
        help="Low-tail cutoffs; condition is q <= cutoff.",
    )
    parser.add_argument(
        "--recompute-q",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On the merged dataset: recompute length-bucket normalization and q "
            "(default: true; required for correct metrics when reading multiple rank*.jsonl)."
        ),
    )
    parser.add_argument(
        "--suffix-len-bin-width",
        type=int,
        default=3,
        help="Must match collection if --recompute-q.",
    )
    parser.add_argument(
        "--suffix-len-min-count",
        type=int,
        default=20,
        help="Must match collection if --recompute-q.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional path to write full result JSON.",
    )
    args = parser.parse_args()

    path_inputs = list(args.jsonl)
    if args.from_dir:
        d = Path(args.from_dir).expanduser().resolve()
        path_inputs.extend(str(p) for p in _paths_from_dir(d, args.glob_name))
    paths = _expand_inputs(path_inputs)

    if not paths:
        print(
            "error: no JSONL files to read.\n"
            f"  --jsonl arguments: {args.jsonl!r}\n"
            f"  --from-dir: {args.from_dir!r}  --glob-name: {args.glob_name!r}\n"
            "Hints:\n"
            "  • If you quoted a glob, Python should still expand it; otherwise the path may be wrong "
            "or filenames may differ (e.g. low_bias_points_rank0.jsonl vs lowtail_...).\n"
            "  • Use: --from-dir /path/to/eval_outcome --glob-name '*rank*.jsonl'\n"
            "  • Or: ls /path/to/eval_outcome/*.jsonl",
            file=sys.stderr,
        )
        return 2

    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    print(f"reading {len(paths)} file(s)", file=sys.stderr)
    for p in paths[:12]:
        print(f"  {p}", file=sys.stderr)
    if len(paths) > 12:
        print(f"  ... and {len(paths) - 12} more", file=sys.stderr)

    records = _load_jsonl_paths(paths)
    if not records:
        nonempty = []
        for p in paths:
            try:
                n = sum(1 for line in p.open("r", encoding="utf-8") if line.strip())
            except OSError as e:
                n = -1
                nonempty.append((p, n, str(e)))
            else:
                nonempty.append((p, n, ""))
        print(
            "error: loaded 0 JSON records. Non-empty line counts per file:",
            file=sys.stderr,
        )
        for p, n, err in nonempty:
            extra = f" ({err})" if err else ""
            print(f"  {p}: {n}{extra}", file=sys.stderr)
        return 2

    if not args.recompute_q and len(paths) > 1:
        print(
            "warning: --no-recompute-q with multiple input files; "
            "shard files usually contain per-rank q — metrics may be wrong. "
            "Omit --no-recompute-q to re-bucket and recompute q on the full union.",
            file=sys.stderr,
        )

    if args.recompute_q:
        _length_bucket_normalize(
            records,
            bin_width=int(args.suffix_len_bin_width),
            min_count=int(args.suffix_len_min_count),
        )
        _assign_q(records, key="f_suffix_rate_norm")
        n_q = sum(
            1
            for r in records
            if np.isfinite(float(r.get("q", float("nan"))))
            and np.isfinite(float(r.get("f_suffix_rate_norm", float("nan"))))
        )
        print(
            "recomputed on full union: "
            f"n_records={len(records)}, finite_q={n_q}, "
            f"suffix_len_bin_width={args.suffix_len_bin_width}, "
            f"suffix_len_min_count={args.suffix_len_min_count}",
            file=sys.stderr,
        )
    else:
        print(
            "using q / f_suffix_rate_norm from JSONL as-is (no recompute).",
            file=sys.stderr,
        )

    tau = float(args.bias_abs_threshold)
    cutoffs = [float(x) for x in args.q_cutoffs]

    # Strong positive-bias set for recall (global, same for all cutoffs)
    strong_pos_mask = []
    for r in records:
        b = float(r.get("bias", float("nan")))
        strong_pos_mask.append(np.isfinite(b) and b > 0.0 and abs(b) > tau)
    strong_pos_idx = np.nonzero(np.asarray(strong_pos_mask))[0]
    n_strong = int(strong_pos_idx.size)

    rows_out: list[dict[str, Any]] = []
    print(f"n_records={len(records)}  n_strong_pos(|bias|>{tau}, bias>0)={n_strong}")
    print(
        "q_cutoff\tn_tail\tbias_pos_in_tail\tP(bias>0|q<=t)\t"
        f"recall_strong_pos_in_tail\tstrong_in_tail\tstrong_total"
    )
    for t in sorted(cutoffs):
        tail_idx = [
            i
            for i, r in enumerate(records)
            if np.isfinite(float(r.get("q", float("nan")))) and float(r["q"]) <= t
        ]
        n_tail = len(tail_idx)
        bias_pos_in_tail = 0
        for i in tail_idx:
            b = float(records[i].get("bias", float("nan")))
            if np.isfinite(b) and b > 0.0:
                bias_pos_in_tail += 1
        p_acc = (bias_pos_in_tail / n_tail) if n_tail > 0 else float("nan")

        strong_in_tail = sum(1 for i in tail_idx if strong_pos_mask[i])
        recall = (strong_in_tail / n_strong) if n_strong > 0 else float("nan")

        row = {
            "q_cutoff": t,
            "n_tail": n_tail,
            "bias_pos_in_tail": bias_pos_in_tail,
            "p_bias_pos_given_q_le_cutoff": p_acc,
            "n_strong_pos_total": n_strong,
            "n_strong_pos_in_tail": strong_in_tail,
            "recall_strong_pos_in_tail": recall,
            "bias_abs_threshold": tau,
        }
        rows_out.append(row)
        print(
            f"{t:g}\t{n_tail}\t{bias_pos_in_tail}\t{p_acc:.6g}\t{recall:.6g}\t{strong_in_tail}\t{n_strong}"
        )

    payload = {
        "inputs": [str(p) for p in paths],
        "recompute_q": bool(args.recompute_q),
        "suffix_len_bin_width": int(args.suffix_len_bin_width),
        "suffix_len_min_count": int(args.suffix_len_min_count),
        "bias_abs_threshold": tau,
        "rows": rows_out,
    }
    if args.out_json:
        outp = Path(args.out_json).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
