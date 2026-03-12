# 数学评估题：10 道题 + 预计算答案，供 test_lora_math 与 test_qwen3_4b_instruct 共用
from typing import List, Tuple

# (题目, 可接受的答案子串列表，模型输出中包含任一即判对)
MATH_EVAL_ITEMS: List[Tuple[str, List[str]]] = [
    ("计算：17 * 24 + 33 等于多少？请给出步骤。", ["441"]),
    ("解方程：2x + 5 = 17，求 x。", ["x=6", "x = 6", "x＝6", "6"]),
    ("一个等差数列首项为 3，公差为 4，求第 10 项。", ["39"]),
    ("化简：(x^2 - 4) / (x - 2)，其中 x ≠ 2。", ["x+2", "x + 2", "x＋2"]),
    ("求函数 f(x) = x^2 - 6x + 9 的最小值及取得最小值时 x 的值。", ["0", "x=3", "x = 3", "最小值0"]),
    ("若 log2(a) = 3，log2(b) = 2，求 log2(a*b) 的值。", ["5"]),
    ("一个圆的半径为 5，求其面积和周长（保留 π）。", ["25π", "25 π", "10π", "10 π"]),
    ("解不等式：3x - 7 > 2x + 4。", ["x>11", "x > 11", "x＞11", "11"]),
    ("从 1 到 100 所有整数的和是多少？请写出计算过程。", ["5050", "5,050"]),
    ("设集合 A = {1, 2, 3}，B = {2, 3, 4}，求 A ∪ B 和 A ∩ B。", ["1,2,3,4", "1, 2, 3, 4", "∪B={1,2,3,4}", "∩B={2,3}", "2,3"]),
]


def check_math_answer(model_output: str, expected_substrings: List[str]) -> bool:
    """模型输出中是否包含任一预期答案子串（忽略首尾空白）。"""
    s = (model_output or "").strip()
    for sub in expected_substrings:
        if sub in s:
            return True
    return False


def run_math_eval(
    questions_with_expected: List[Tuple[str, List[str]]],
    generate_fn,  # (question: str) -> str
) -> Tuple[int, int, List[bool]]:
    """
    运行数学题评估。generate_fn(question) 返回模型输出字符串。
    返回 (正确数, 总数, 每题是否正确的列表)。
    """
    results: List[bool] = []
    for q, expected in questions_with_expected:
        out = generate_fn(q)
        results.append(check_math_answer(out, expected))
    correct = sum(results)
    return correct, len(results), results
