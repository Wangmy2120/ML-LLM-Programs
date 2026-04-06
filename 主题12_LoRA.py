# 主题 12：LoRA（Low-Rank Adaptation）
# 记忆点：冻结原权重 W，仅训练低秩增量 DeltaW = B @ A。

from typing import List

from common_utils import matmul

Matrix = List[List[float]]


def lora_forward(x: Matrix, w: Matrix, a: Matrix, b: Matrix, alpha: float, r: int) -> Matrix:
    """y = x @ (W + alpha/r * (B@A))"""
    # 先构造低秩增量权重 DeltaW，再按 alpha/r 缩放后与原权重相加。
    delta_w = matmul(b, a)
    scaled_delta_w = [[(alpha / r) * value for value in row] for row in delta_w]
    merged_w = [[w[i][j] + scaled_delta_w[i][j] for j in range(len(w[0]))] for i in range(len(w))]
    return matmul(x, merged_w)


def demo() -> None:
    # x: [batch=2, in_dim=3]
    x = [[1.0, 2.0, 1.0], [0.5, 1.0, 1.5]]
    # W: [in_dim=3, out_dim=2]
    w = [[0.2, 0.1], [0.0, 0.3], [0.4, 0.2]]

    # 低秩分解：B[in_dim, r], A[r, out_dim], r=1
    b = [[0.1], [0.2], [0.3]]
    a = [[0.5, -0.5]]

    y = lora_forward(x, w, a, b, alpha=2.0, r=1)
    print("LoRA 输出:")
    for row in y:
        print([round(v, 4) for v in row])

    print("复杂度: 训练参数从 O(in*out) 降到 O(r*(in+out))")


if __name__ == "__main__":
    demo()

