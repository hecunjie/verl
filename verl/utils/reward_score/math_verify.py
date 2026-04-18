# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass

    return ret_score


def compute_score_with_pred(
    model_output: str, ground_truth: str, timeout_score: float = 0
) -> tuple[float, str | None]:
    """Same scoring as ``compute_score``, plus the extracted prediction string from Math-Verify.

    The second return of ``math_metric`` is ``(golds, preds)`` flattened string lists; we surface
    ``preds`` (joined if multiple) as the prediction text used for verification.
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    str_preds: tuple[list[str], list[str]] | None = None

    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, str_preds = verify_func([ground_truth_boxed], [model_output])
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass

    pred_str: str | None = None
    if str_preds is not None:
        _golds, preds = str_preds
        if preds:
            pred_str = ", ".join(preds) if len(preds) > 1 else preds[0]

    return ret_score, pred_str
