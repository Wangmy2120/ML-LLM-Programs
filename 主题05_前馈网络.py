# 主题 5：Feed Forward Network（前馈网络 / MLP）
# Transformer 中每个 token 都会独立经过同一个两层前馈网络
# 常见结构：FFN(x) = W2 * activation(W1 * x + b1) + b2
# 面试记忆点：
# 1. 对每个 token 独立处理，不发生 token 间交互
# 2. 通常先升维，再激活，再降维
# 3. 常见激活函数有 ReLU、GELU

from typing import List


Vector = List[float]
Matrix = List[List[float]]


def linear(x: Vector, weight: Matrix, bias: Vector) -> Vector:
    """最小线性层：y = xW + b。"""
    result = []
    for out_index in range(len(weight[0])):
        value = bias[out_index]
        for in_index in range(len(x)):
            value += x[in_index] * weight[in_index][out_index]
        result.append(value)
    return result


def relu(x: Vector) -> Vector:
    """ReLU 激活：负数截断为 0。"""
    return [max(0.0, value) for value in x]


def feed_forward(token: Vector, w1: Matrix, b1: Vector, w2: Matrix, b2: Vector) -> Vector:
    """单 token 的两层前馈：Linear -> ReLU -> Linear。"""
    hidden = linear(token, w1, b1)
    hidden = relu(hidden)
    return linear(hidden, w2, b2)


def feed_forward_batch(x: Matrix, w1: Matrix, b1: Vector, w2: Matrix, b2: Vector) -> Matrix:
    """批量版本：对每个 token 独立应用同一个 FFN。"""
    return [feed_forward(token, w1, b1, w2, b2) for token in x]


def demo() -> None:
    x = [
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.0, 1.5, 2.0],
    ]
    w1 = [
        [0.1, 0.2, 0.3],
        [0.0, 0.1, 0.0],
        [0.2, 0.1, 0.2],
        [0.1, 0.0, 0.1],
    ]
    b1 = [0.1, 0.1, 0.1]
    w2 = [
        [0.2, 0.1, 0.0, 0.1],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.1],
    ]
    b2 = [0.0, 0.0, 0.0, 0.0]

    output = feed_forward_batch(x, w1, b1, w2, b2)
    print("FFN 输出:")
    for row in output:
        print([round(value, 4) for value in row])
    print("复杂度: 单层 token 数为 n、维度为 d 时，约为 O(n * d^2)")


if __name__ == "__main__":
    demo()

