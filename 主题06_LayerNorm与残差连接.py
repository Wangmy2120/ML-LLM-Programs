# 主题 6：LayerNorm 与 Residual（层归一化与残差连接）
# 面试记忆点：
# 1. LayerNorm 是按单个 token 的特征维做归一化
# 2. 残差连接是 output = x + sublayer(x)
# 3. Transformer 常见为 Pre-LN：x -> LN -> 子层 -> 残差相加

from typing import List

from common_utils import layer_norm


Vector = List[float]
Matrix = List[List[float]]


def add_residual(x: Matrix, sublayer_output: Matrix) -> Matrix:
    return [
        [x[i][j] + sublayer_output[i][j] for j in range(len(x[i]))]
        for i in range(len(x))
    ]


def pre_norm_residual(x: Matrix, sublayer_output: Matrix) -> Matrix:
    normalized = [layer_norm(token) for token in x]
    return add_residual(normalized, sublayer_output)


def demo() -> None:
    x = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 2.0, 2.0, 2.0],
    ]
    sublayer_output = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.4, 0.3, 0.2],
    ]

    print("LayerNorm 结果:")
    for row in x:
        print([round(value, 4) for value in layer_norm(row)])

    print("Pre-LN + Residual 结果:")
    output = pre_norm_residual(x, sublayer_output)
    for row in output:
        print([round(value, 4) for value in row])

    print("复杂度: 时间 O(seq_len * d_model)，空间 O(d_model)")


if __name__ == "__main__":
    demo()
