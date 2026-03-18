# 主题 7：Transformer Block（一个基础 Transformer 块）
# 结构主线：
# x -> LayerNorm -> Self-Attention -> 残差相加
#   -> LayerNorm -> FFN -> 残差相加
# 面试记忆点：
# 1. 注意力负责 token 间信息交互
# 2. FFN 负责 token 内非线性变换
# 3. Add & Norm 是稳定训练的关键结构
# 说明：这里用最小可运行实现，便于手撕和讲解流程

import math
from typing import List


Vector = List[float]
Matrix = List[List[float]]


def stable_softmax(logits: List[float]) -> List[float]:
    max_value = max(logits)
    exp_values = [math.exp(value - max_value) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            value = 0.0
            for k in range(len(b)):
                value += a[i][k] * b[k][j]
            row.append(value)
        result.append(row)
    return result


def layer_norm(token: Vector, eps: float = 1e-5) -> Vector:
    avg = sum(token) / len(token)
    var = sum((value - avg) ** 2 for value in token) / len(token)
    std = math.sqrt(var + eps)
    return [(value - avg) / std for value in token]


def add_matrix(a: Matrix, b: Matrix) -> Matrix:
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(a[i])):
            row.append(a[i][j] + b[i][j])
        result.append(row)
    return result


def self_attention(x: Matrix) -> Matrix:
    scores = matmul(x, transpose(x))
    scale = math.sqrt(len(x[0]))
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            scores[i][j] /= scale
    weights = [stable_softmax(row) for row in scores]
    return matmul(weights, x)


def linear(x: Vector, weight: Matrix, bias: Vector) -> Vector:
    result = []
    for out_index in range(len(weight[0])):
        value = bias[out_index]
        for in_index in range(len(x)):
            value += x[in_index] * weight[in_index][out_index]
        result.append(value)
    return result


def relu(x: Vector) -> Vector:
    return [max(0.0, value) for value in x]


def feed_forward(x: Matrix) -> Matrix:
    d_model = len(x[0])
    hidden_dim = d_model + 1
    w1 = [[0.1 for _ in range(hidden_dim)] for _ in range(d_model)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [[0.1 for _ in range(d_model)] for _ in range(hidden_dim)]
    b2 = [0.0 for _ in range(d_model)]

    output = []
    for token in x:
        hidden = relu(linear(token, w1, b1))
        output.append(linear(hidden, w2, b2))
    return output


def transformer_block(x: Matrix) -> Matrix:
    norm1 = [layer_norm(token) for token in x]
    attn_output = self_attention(norm1)
    x = add_matrix(x, attn_output)

    norm2 = [layer_norm(token) for token in x]
    ffn_output = feed_forward(norm2)
    x = add_matrix(x, ffn_output)
    return x


def demo() -> None:
    x = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]
    output = transformer_block(x)
    print("Transformer Block 输出:")
    for row in output:
        print([round(value, 4) for value in row])
    print("复杂度: 注意力部分主导，约为 O(seq_len^2 * d_model)")


if __name__ == "__main__":
    demo()

