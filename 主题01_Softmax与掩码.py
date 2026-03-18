# 主题 1：Softmax 与 Mask（大模型 / Transformer 基础手撕）
# 适用场景：注意力机制、分类概率、采样前归一化
# 核心公式：softmax(x_i) = exp(x_i) / sum(exp(x_j))
# 稳定写法：先减去最大值，避免 exp 溢出
# 面试记忆点：
# 1. softmax 要做数值稳定处理
# 2. mask 本质是把不该看的位置变成极小值
# 3. causal mask 用于防止看到未来 token

from typing import List

from common_utils import stable_softmax


def build_causal_mask(seq_len: int) -> List[List[float]]:
    """构造下三角 mask，可看位置为 0，不可看位置为负无穷。"""
    return [
        [0.0 if col <= row else float("-inf") for col in range(seq_len)]
        for row in range(seq_len)
    ]


def apply_mask(scores: List[List[float]], mask: List[List[float]]) -> List[List[float]]:
    """将 mask 加到分数矩阵上。"""
    return [
        [scores[i][j] + mask[i][j] for j in range(len(scores[i]))]
        for i in range(len(scores))
    ]


def demo() -> None:
    logits = [2.0, 1.0, 0.1]
    probs = stable_softmax(logits)
    print("softmax 输入:", logits)
    print("softmax 输出:", [round(value, 4) for value in probs])

    scores = [
        [1.2, 0.8, 0.3],
        [0.5, 1.1, 0.9],
        [0.2, 0.4, 1.5],
    ]
    mask = build_causal_mask(seq_len=3)
    masked_scores = apply_mask(scores, mask)

    print("causal mask:")
    for row in mask:
        print(row)

    print("masked scores:")
    for row in masked_scores:
        print(row)

    print("复杂度: softmax 为 O(n)，mask 作用到矩阵为 O(n^2)")


if __name__ == "__main__":
    demo()

