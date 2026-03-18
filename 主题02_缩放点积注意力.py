# 主题 2：Scaled Dot-Product Attention（缩放点积注意力）
# 核心公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
# 面试记忆点：
# 1. 先算相似度分数 QK^T
# 2. 再除以 sqrt(d_k) 防止分数过大
# 3. softmax 后得到注意力权重
# 4. 权重与 V 加权求和得到输出

import math
from typing import List, Optional, Tuple

from common_utils import matmul, row_softmax, transpose


Matrix = List[List[float]]


def apply_mask(scores: Matrix, mask: Optional[Matrix]) -> Matrix:
    if mask is None:
        return scores
    return [
        [scores[i][j] + mask[i][j] for j in range(len(scores[i]))]
        for i in range(len(scores))
    ]


def scaled_dot_product_attention(
    q: Matrix,
    k: Matrix,
    v: Matrix,
    mask: Optional[Matrix] = None,
) -> Tuple[Matrix, Matrix]:
    """返回注意力输出和注意力权重。"""
    dk = len(k[0])
    scores = matmul(q, transpose(k))

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            scores[i][j] /= math.sqrt(dk)

    scores = apply_mask(scores, mask)
    weights = row_softmax(scores)
    output = matmul(weights, v)
    return output, weights


def demo() -> None:
    q = [[1.0, 0.0], [0.0, 1.0]]
    k = [[1.0, 0.0], [0.0, 1.0]]
    v = [[10.0, 1.0], [1.0, 10.0]]

    output, weights = scaled_dot_product_attention(q, k, v)
    print("attention 权重:")
    for row in weights:
        print([round(value, 4) for value in row])

    print("attention 输出:")
    for row in output:
        print([round(value, 4) for value in row])

    print("复杂度: 时间 O(seq_len^2 * d)，空间 O(seq_len^2)")


if __name__ == "__main__":
    demo()
