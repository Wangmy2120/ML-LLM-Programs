# 主题 3：Multi-Head Attention（多头注意力）
# 核心思想：把 d_model 切成多个 head，每个 head 学不同子空间的信息，再拼回去
# 面试记忆点：
# 1. 线性投影得到 Q、K、V
# 2. split heads: [seq_len, d_model] -> [num_heads, seq_len, head_dim]
# 3. 每个 head 单独做 attention
# 4. merge heads: [num_heads, seq_len, head_dim] -> [seq_len, d_model]
# 说明：这里为了便于背诵，省略线性层，只演示多头拆分与合并主流程

import math
from typing import List, Tuple

from common_utils import matmul, stable_softmax, transpose


Matrix = List[List[float]]
Tensor3D = List[Matrix]


def single_head_attention(q: Matrix, k: Matrix, v: Matrix) -> Tuple[Matrix, Matrix]:
    scores = matmul(q, transpose(k))
    scale = math.sqrt(len(k[0]))
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            scores[i][j] /= scale
    weights = [stable_softmax(row) for row in scores]
    output = matmul(weights, v)
    return output, weights


def split_heads(x: Matrix, num_heads: int) -> Tensor3D:
    seq_len = len(x)
    d_model = len(x[0])
    if d_model % num_heads != 0:
        raise ValueError("d_model 必须能被 num_heads 整除")

    head_dim = d_model // num_heads
    heads = []
    for head_index in range(num_heads):
        head = []
        start = head_index * head_dim
        end = start + head_dim
        for token in range(seq_len):
            head.append(x[token][start:end])
        heads.append(head)
    return heads


def merge_heads(heads: Tensor3D) -> Matrix:
    seq_len = len(heads[0])
    result = []
    for token in range(seq_len):
        merged = []
        for head in heads:
            merged.extend(head[token])
        result.append(merged)
    return result


def multi_head_attention(x: Matrix, num_heads: int) -> Matrix:
    """为便于手撕记忆，这里直接使用 x 作为 Q/K/V。"""
    q_heads = split_heads(x, num_heads)
    k_heads = split_heads(x, num_heads)
    v_heads = split_heads(x, num_heads)

    output_heads = []
    for q_head, k_head, v_head in zip(q_heads, k_heads, v_heads):
        output, _ = single_head_attention(q_head, k_head, v_head)
        output_heads.append(output)

    return merge_heads(output_heads)


def demo() -> None:
    x = [
        [1.0, 0.0, 2.0, 1.0],
        [0.0, 1.0, 1.0, 2.0],
        [1.0, 1.0, 0.0, 1.0],
    ]
    output = multi_head_attention(x, num_heads=2)
    print("MHA 输出:")
    for row in output:
        print([round(value, 4) for value in row])
    print("复杂度: 时间 O(num_heads * seq_len^2 * head_dim)")


if __name__ == "__main__":
    demo()
