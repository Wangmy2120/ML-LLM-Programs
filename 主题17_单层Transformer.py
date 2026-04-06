# 主题 17：单层 Transformer（简易手撕版）
# 结构：x -> LN -> Self-Attention -> 残差
#      -> LN -> FFN -> 残差
# 说明：为便于手撕记忆，这里直接使用 x 作为 Q/K/V，不引入额外线性投影层。

import math
from typing import List

from common_utils import layer_norm, matmul, stable_softmax, transpose

Vector = List[float]
Matrix = List[List[float]]


def add_matrix(a: Matrix, b: Matrix) -> Matrix:
    """两个同形状矩阵逐元素相加。"""
    return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]


def self_attention(x: Matrix) -> Matrix:
    """简化自注意力：Q=K=V=x。"""
    scores = matmul(x, transpose(x))
    scale = math.sqrt(len(x[0]))
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            scores[i][j] /= scale
    weights = [stable_softmax(row) for row in scores]
    return matmul(weights, x)


def linear(x: Vector, weight: Matrix, bias: Vector) -> Vector:
    """线性层 y = xW + b。"""
    result: Vector = []
    for out_index in range(len(weight[0])):
        value = bias[out_index]
        for in_index in range(len(x)):
            value += x[in_index] * weight[in_index][out_index]
        result.append(value)
    return result


def relu(x: Vector) -> Vector:
    """ReLU 激活。"""
    return [max(0.0, value) for value in x]


def feed_forward(x: Matrix, hidden_dim: int | None = None) -> Matrix:
    """简化 FFN：Linear -> ReLU -> Linear。"""
    d_model = len(x[0])
    # 采用 Transformer 常见配置：FFN 隐层维度约为 4 * d_model。
    hidden_dim = hidden_dim or (4 * d_model)
    # 这里每次前向都构造固定权重，仅用于手撕演示流程，不涉及参数学习。
    w1 = [[0.1 for _ in range(hidden_dim)] for _ in range(d_model)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [[0.1 for _ in range(d_model)] for _ in range(hidden_dim)]
    b2 = [0.0 for _ in range(d_model)]
    return [linear(relu(linear(token, w1, b1)), w2, b2) for token in x]


def single_layer_transformer(x: Matrix) -> Matrix:
    """单层 Transformer：Pre-LN Attention + Pre-LN FFN。"""
    norm1 = [layer_norm(token) for token in x]
    attn_out = self_attention(norm1)
    x = add_matrix(x, attn_out)

    norm2 = [layer_norm(token) for token in x]
    ffn_out = feed_forward(norm2)
    x = add_matrix(x, ffn_out)
    return x


def demo() -> None:
    x = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]
    output = single_layer_transformer(x)
    print("单层 Transformer 输出:")
    for row in output:
        print([round(value, 4) for value in row])
    print("复杂度: 注意力主导，约 O(seq_len^2 * d_model)")


if __name__ == "__main__":
    demo()
