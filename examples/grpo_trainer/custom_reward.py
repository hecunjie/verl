import re

def compute_score(solution_str, ground_truth, method='strict', format='score', **kwargs):
    """
    自定义奖励函数示例。
    Args:
        solution_str: 模型生成的思维链和答案。
        ground_truth: 期望的正确答案。
        method: 评分方法（可选）。
        format: 返回格式（可选）。
    """
    # 示例逻辑：简单检查 ground_truth 是否包含在 solution_str 中
    # 您可以根据实际需求修改此处的逻辑，例如解析 XML 标签、数学公式匹配等
    if str(ground_truth) in solution_str:
        return 1.0
    return 0.0
