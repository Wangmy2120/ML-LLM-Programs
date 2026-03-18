"""llm_ACM 公共工具函数。

目标：把高频基础操作收敛到一个文件，减少各题重复代码，方便背诵。
仅依赖 Python 标准库。
"""

import math
from typing import List

Vector = List[float]
Matrix = List[List[float]]


def stable_softmax(logits: Vector) -> Vector:
    """数值稳定 softmax：先减最大值，避免 exp 溢出。"""
    max_value = max(logits)
    exp_values = [math.exp(value - max_value) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def transpose(matrix: Matrix) -> Matrix:
    """矩阵转置。"""
    return [list(column) for column in zip(*matrix)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """最小矩阵乘法实现。"""
    rows = len(a)
    cols = len(b[0])
    mid = len(b)
    result: Matrix = []
    for i in range(rows):
        row: Vector = []
        for j in range(cols):
            value = 0.0
            for k in range(mid):
                value += a[i][k] * b[k][j]
            row.append(value)
        result.append(row)
    return result


def dot(a: Vector, b: Vector) -> float:
    """向量点积。"""
    return sum(x * y for x, y in zip(a, b))


def layer_norm(token: Vector, eps: float = 1e-5) -> Vector:
    """不带可学习参数的 LayerNorm。"""
    avg = sum(token) / len(token)
    var = sum((value - avg) ** 2 for value in token) / len(token)
    std = math.sqrt(var + eps)
    return [(value - avg) / std for value in token]


def rms_norm(token: Vector, eps: float = 1e-5) -> Vector:
    """不减均值的 RMSNorm。"""
    rms = math.sqrt(sum(value * value for value in token) / len(token) + eps)
    return [value / rms for value in token]


def row_softmax(matrix: Matrix) -> Matrix:
    """按行做 softmax。"""
    return [stable_softmax(row) for row in matrix]

