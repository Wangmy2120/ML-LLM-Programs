# 主题 13：LayerNorm 与 RMSNorm
# 记忆点：
# 1) LayerNorm: 减均值 + 除标准差。
# 2) RMSNorm: 只除 RMS，不减均值，计算更简。

from typing import List

from common_utils import layer_norm, rms_norm


Vector = List[float]


def demo() -> None:
    x: Vector = [1.0, 2.0, 3.0, 4.0]
    ln = layer_norm(x)
    rmsn = rms_norm(x)

    print("输入:", x)
    print("LayerNorm:", [round(v, 4) for v in ln])
    print("RMSNorm:", [round(v, 4) for v in rmsn])
    print("复杂度: 两者均为 O(d)")


if __name__ == "__main__":
    demo()

