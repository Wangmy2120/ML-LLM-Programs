# 主题 11：MHA / GQA
# 记忆点：
# 1) MHA: 每个 Q 头对应一个 K/V 头。
# 2) GQA: 多个 Q 头共享同一个 K/V 头，降低 KV cache 成本。

import math
from typing import List

from common_utils import matmul, stable_softmax, transpose

Matrix = List[List[float]]
Tensor3D = List[Matrix]


def split_heads(x: Matrix, num_heads: int) -> Tensor3D:
    """按特征维拆分多头：[seq_len, d_model] -> [num_heads, seq_len, head_dim]。"""
    seq_len, d_model = len(x), len(x[0])
    if d_model % num_heads != 0:
        raise ValueError("d_model 必须能被 num_heads 整除")
    head_dim = d_model // num_heads
    return [[x[t][h * head_dim:(h + 1) * head_dim] for t in range(seq_len)] for h in range(num_heads)]


def merge_heads(heads: Tensor3D) -> Matrix:
    """合并多头回原维度：[num_heads, seq_len, head_dim] -> [seq_len, d_model]。"""
    seq_len = len(heads[0])
    return [[value for head in heads for value in head[t]] for t in range(seq_len)]


def attention(q: Matrix, k: Matrix, v: Matrix) -> Matrix:
    """单头缩放点积注意力。"""
    scores = matmul(q, transpose(k))
    scale = math.sqrt(len(k[0]))
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            scores[i][j] /= scale
    weights = [stable_softmax(row) for row in scores]
    return matmul(weights, v)


def mha(x: Matrix, num_heads: int) -> Matrix:
    """标准 MHA：每个 Q 头使用对应的 K/V 头。"""
    q_heads = split_heads(x, num_heads)
    k_heads = split_heads(x, num_heads)
    v_heads = split_heads(x, num_heads)
    out_heads = [attention(q, k, v) for q, k, v in zip(q_heads, k_heads, v_heads)]
    return merge_heads(out_heads)


def gqa(x: Matrix, num_heads: int, num_kv_heads: int) -> Matrix:
    """GQA：将多个 Q 头分组后共享同一组 K/V 头。"""
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads 必须是 num_kv_heads 的整数倍")

    # 为了保证手撕示例维度一致，先按 num_heads 切分，再按组共享 K/V 头。
    q_heads = split_heads(x, num_heads)
    full_k_heads = split_heads(x, num_heads)
    full_v_heads = split_heads(x, num_heads)

    group_size = num_heads // num_kv_heads
    shared_k_heads = [full_k_heads[i * group_size] for i in range(num_kv_heads)]
    shared_v_heads = [full_v_heads[i * group_size] for i in range(num_kv_heads)]

    out_heads: Tensor3D = []
    for i, q in enumerate(q_heads):
        kv_idx = i // group_size  # 一组 Q 头共享一个 K/V 头
        out_heads.append(attention(q, shared_k_heads[kv_idx], shared_v_heads[kv_idx]))

    return merge_heads(out_heads)


def demo() -> None:
    x = [
        [1.0, 0.0, 2.0, 1.0],
        [0.0, 1.0, 1.0, 2.0],
        [1.0, 1.0, 0.0, 1.0],
    ]
    print("MHA:")
    for row in mha(x, num_heads=2):
        print([round(v, 4) for v in row])

    print("GQA(num_heads=2, num_kv_heads=1):")
    for row in gqa(x, num_heads=2, num_kv_heads=1):
        print([round(v, 4) for v in row])

    print("复杂度: MHA 约 O(h * n^2 * d_h)，GQA 减少 K/V 头后更省缓存")


if __name__ == "__main__":
    demo()

