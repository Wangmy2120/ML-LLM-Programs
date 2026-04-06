# 主题 8：KV Cache（大模型推理常考）
# 面试记忆点：
# 1. 自回归生成时，历史 token 的 K / V 不需要重复计算
# 2. 每次只算新 token 的 Q，然后与缓存的 K / V 做注意力
# 3. KV Cache 可以把重复计算从“整句重算”变成“增量追加”

import math
from typing import List, Tuple

from common_utils import dot, stable_softmax


Vector = List[float]
Matrix = List[List[float]]


class KVCache:
    """最小 KV Cache 实现。"""

    def __init__(self) -> None:
        self.keys = []
        self.values = []

    def append(self, key: Vector, value: Vector) -> None:
        self.keys.append(key)
        self.values.append(value)

    def size(self) -> int:
        return len(self.keys)


def weighted_sum(weights: Vector, values: Matrix) -> Vector:
    """按注意力权重对 value 向量序列做加权求和。"""
    result = [0.0 for _ in range(len(values[0]))]
    for weight, value in zip(weights, values):
        for index in range(len(value)):
            result[index] += weight * value[index]
    return result


def decode_step(query: Vector, key: Vector, value: Vector, cache: KVCache) -> Tuple[Vector, Vector]:
    """单步解码：先把新 token 的 K/V 放入缓存，再只用当前 Q 与所有缓存做注意力。"""
    cache.append(key, value)
    # 缩放点积得到 query 对历史所有 key 的打分。
    scale = math.sqrt(len(query))
    scores = [dot(query, cached_key) / scale for cached_key in cache.keys]
    weights = stable_softmax(scores)
    output = weighted_sum(weights, cache.values)
    return output, weights


def demo() -> None:
    cache = KVCache()

    step_1_output, step_1_weights = decode_step(
        query=[1.0, 0.0],
        key=[1.0, 0.0],
        value=[10.0, 1.0],
        cache=cache,
    )
    print("step1 cache size:", cache.size())
    print("step1 weights:", [round(value, 4) for value in step_1_weights])
    print("step1 output:", [round(value, 4) for value in step_1_output])

    step_2_output, step_2_weights = decode_step(
        query=[0.0, 1.0],
        key=[0.0, 1.0],
        value=[1.0, 10.0],
        cache=cache,
    )
    print("step2 cache size:", cache.size())
    print("step2 weights:", [round(value, 4) for value in step_2_weights])
    print("step2 output:", [round(value, 4) for value in step_2_output])

    print("复杂度: 单步注意力计算约为 O(t * d)，其中 t 为当前缓存长度")


if __name__ == "__main__":
    demo()
