# 主题 4：Positional Encoding（位置编码）
# 核心公式：
# PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
# 面试记忆点：
# 1. Transformer 本身不感知顺序，需要额外加位置编码
# 2. 偶数维用 sin，奇数维用 cos
# 3. 不同频率帮助模型表达远近位置信息

import math
from typing import List


Matrix = List[List[float]]


def positional_encoding(seq_len: int, d_model: int) -> Matrix:
    """生成正弦位置编码矩阵，形状为 [seq_len, d_model]。"""
    if d_model <= 0:
        raise ValueError("d_model 必须大于 0")

    result = []
    for pos in range(seq_len):
        row = []
        for i in range(d_model):
            # i//2 让偶数/奇数维共享同一频率，只在 sin/cos 上区分。
            angle_rate = pos / (10000 ** (2 * (i // 2) / d_model))
            if i % 2 == 0:
                row.append(math.sin(angle_rate))
            else:
                row.append(math.cos(angle_rate))
        result.append(row)
    return result


def demo() -> None:
    pe = positional_encoding(seq_len=4, d_model=6)
    print("位置编码前 4 行:")
    for row in pe:
        print([round(value, 4) for value in row])
    print("复杂度: 时间 O(seq_len * d_model)，空间 O(seq_len * d_model)")


if __name__ == "__main__":
    demo()

