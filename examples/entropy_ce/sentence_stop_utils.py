"""按「第一句结束」停止 vLLM 续写：自定义停止逻辑，避免把英文小数点当句末。

vLLM 的 ``SamplingParams(stop=...)`` 只能做子串匹配，无法区分 ``3.14`` 与句末 ``.``。
本模块用 **分块 generate**（每块最多 ``chunk_max_tokens`` token），每块后 **decode 已生成后缀**
再判断是否应停；熵沿 **实际保留的 token** 累加。

可选：``pip install pysbd`` 后使用 ``make_pysbd_first_sentence_stop_check()`` 得到更稳的英文句界。

使用示例
--------

.. code-block:: python

    from vllm import SamplingParams
    from entropy_credit_experiment import clamp_vllm_logprobs_topk, entropy_from_logprobs_topk
    from sentence_stop_utils import generate_until_sentence_boundary_vllm

    k = clamp_vllm_logprobs_topk(20)

    def sp_factory(chunk: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=chunk,
            temperature=1.0,
            top_p=0.95,
            logprobs=k,
        )

    gen_ids, scores, by_sent = generate_until_sentence_boundary_vllm(
        llm, tokenizer, prefix_token_ids, sp_factory,
        chunk_max_tokens=32,
        max_total_new_tokens=256,
    )
    F_short = sum(entropy_from_logprobs_topk(s) for s in scores)
"""

from __future__ import annotations

import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 句末检测（仅看「续写后缀」文本，不含 prompt）
# ---------------------------------------------------------------------------


def _ends_with_chinese_sentence_punct(s: str) -> bool:
    t = s.rstrip()
    return bool(t) and t[-1] in "。！？"


def _likely_decimal_or_incomplete_number_at_end(s: str) -> bool:
    """末尾的 ``.`` 若像小数或未写完的数字，则不当句末。"""
    t = s.rstrip()
    if not t or t[-1] != ".":
        return False
    if re.search(r"\d\.\s*$", t):
        return True
    if re.search(r"\d\.\d+\s*$", t):
        return True
    return False


def _ends_with_english_sentence_punct_simple(s: str) -> bool:
    """启发式：``!`` ``?`` 或 ``.``（且非小数启发式）。"""
    t = s.rstrip()
    if not t:
        return False
    if t[-1] in "!?":
        return True
    if t[-1] != ".":
        return False
    if _likely_decimal_or_incomplete_number_at_end(t):
        return False
    return True


def _segment_with_pysbd(text: str, language: str = "en") -> list[str]:
    try:
        import pysbd  # type: ignore[import-untyped]
    except ImportError:
        return []
    seg = pysbd.Segmenter(language=language, clean=False)
    return list(seg.segment(text.strip()))


def completion_should_stop_after_first_sentence_simple(
    completion_text: str,
    *,
    min_chars: int = 2,
) -> bool:
    """默认停止条件：中文 ``。！？``；英文 ``.!?`` + 小数末尾启发式（不依赖 pysbd）。"""
    t = completion_text
    if len(t.strip()) < min_chars:
        return False
    if _ends_with_chinese_sentence_punct(t):
        return True
    return _ends_with_english_sentence_punct_simple(t)


def make_pysbd_first_sentence_stop_check(
    *,
    language: str = "en",
    min_chars: int = 2,
) -> Callable[[str], bool]:
    """返回 ``stop_check(text)``：用 pysbd 判断「首句是否已完整出现」（适合英文为主）。

    若未安装 pysbd，等价于 ``completion_should_stop_after_first_sentence_simple``。
    """

    def stop_check(completion_text: str) -> bool:
        t = completion_text
        if len(t.strip()) < min_chars:
            return False
        if _ends_with_chinese_sentence_punct(t):
            return True
        sents = _segment_with_pysbd(t, language)
        if not sents:
            return _ends_with_english_sentence_punct_simple(t)
        first = sents[0].strip()
        if len(first) < min_chars:
            return False
        if first[-1] not in ".!?…":
            return False
        # 续写已至少包含完整首句（允许首句后多一两个空白）
        ts = t.strip()
        if ts.startswith(first) and len(ts) <= len(first) + 2:
            return True
        # 或整体已以句末标点结束且 pysbd 切出首句（长 CoT 首句可能很长）
        if ts.endswith(first[-1]) and first in ts:
            return len(sents) == 1 or len(ts) >= len(first)
        return _ends_with_english_sentence_punct_simple(t)

    return stop_check


# ---------------------------------------------------------------------------
# vLLM：分块续写直到句末（或上限）
# ---------------------------------------------------------------------------


def generate_until_sentence_boundary_vllm(
    llm: Any,
    tokenizer: Any,
    prefix_token_ids: list[int],
    sampling_params_factory: Callable[[int], Any],
    *,
    chunk_max_tokens: int = 48,
    max_total_new_tokens: int = 512,
    stop_check: Callable[[str], bool] | None = None,
    quiet_generate: Callable[[Any, list, Any], Any] | None = None,
) -> tuple[list[int], list[dict[int, float]], bool]:
    """从 ``prefix_token_ids`` 起分块生成，直到 ``stop_check(续写文本)`` 或长度/EOS。

    Parameters
    ----------
    sampling_params_factory
        ``(chunk_max_tokens: int) -> SamplingParams``，由调用方填入 ``temperature/top_p/logprobs`` 等；
        本函数用当前 chunk 的 token 上限调用它。
    stop_check
        对 **仅续写** 的 decode 字符串判断；默认 ``completion_should_stop_after_first_sentence_simple``。
    quiet_generate
        默认 ``entropy_credit_experiment.vllm_generate_quiet``。

    Returns
    -------
    completion_token_ids
        新生成的 token id 列表（不含 prefix）。
    logprobs_per_step
        与每步生成对齐的 top-k logprob 字典列表。
    stopped_by_sentence
        ``True``：因句末逻辑停止；``False``：触达 ``max_total_new_tokens``、空生成或 EOS。
    """
    from vllm.inputs import TokensPrompt

    if quiet_generate is None:
        from entropy_credit_experiment import vllm_generate_quiet as quiet_generate

    if stop_check is None:
        stop_check = completion_should_stop_after_first_sentence_simple

    cur = list(prefix_token_ids)
    completion_ids: list[int] = []
    logprobs_per_step: list[dict[int, float]] = []
    stopped_by_sentence = False
    budget = int(max_total_new_tokens)

    while budget > 0:
        step_chunk = min(int(chunk_max_tokens), budget)
        sp = sampling_params_factory(step_chunk)
        outputs = quiet_generate(llm, [TokensPrompt(prompt_token_ids=cur)], sp)
        o = outputs[0].outputs[0]
        gen_ids = list(o.token_ids)
        if not gen_ids:
            break

        for step_lp in o.logprobs or []:
            d: dict[int, float] = {}
            for tid, info in step_lp.items():
                d[int(tid)] = float(info.logprob)
            logprobs_per_step.append(d)

        completion_ids.extend(gen_ids)
        while len(logprobs_per_step) < len(completion_ids):
            logprobs_per_step.append({})

        cur = cur + gen_ids
        budget -= len(gen_ids)

        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        if stop_check(completion_text):
            stopped_by_sentence = True
            break

    return completion_ids, logprobs_per_step[: len(completion_ids)], stopped_by_sentence
