# 主题 14：熵、KL 散度、交叉熵
# 记忆点：
# - 熵 H(P): 分布自身不确定性。
# - KL(P||Q): 用 Q 近似 P 的信息损失。
# - 交叉熵 CE(P,Q) = H(P) + KL(P||Q)。

import math
from typing import List

Vector = List[float]


def entropy(p: Vector, eps: float = 1e-12) -> float:
    return -sum(pi * math.log(pi + eps) for pi in p)


def kl_divergence(p: Vector, q: Vector, eps: float = 1e-12) -> float:
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def cross_entropy(p: Vector, q: Vector, eps: float = 1e-12) -> float:
    return -sum(pi * math.log(qi + eps) for pi, qi in zip(p, q))


def demo() -> None:
    p = [0.7, 0.2, 0.1]
    q = [0.6, 0.3, 0.1]

    h = entropy(p)
    kl = kl_divergence(p, q)
    ce = cross_entropy(p, q)

    print(f"H(P)={h:.6f}")
    print(f"KL(P||Q)={kl:.6f}")
    print(f"CE(P,Q)={ce:.6f}")
    print(f"验证 CE≈H+KL: {ce:.6f} vs {(h + kl):.6f}")
    print("复杂度: O(n)")


if __name__ == "__main__":
    demo()

