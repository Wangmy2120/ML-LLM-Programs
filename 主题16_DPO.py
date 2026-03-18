# 主题 16：DPO（Direct Preference Optimization）
# 记忆点：
# 1) 不需要奖励模型，直接用偏好对 (chosen, rejected) 训练。
# 2) 目标让策略在 chosen 上相对 rejected 的对数概率差更大。

import math
from typing import List


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def dpo_loss(
    pi_logp_chosen: List[float],
    pi_logp_rejected: List[float],
    ref_logp_chosen: List[float],
    ref_logp_rejected: List[float],
    beta: float = 0.1,
) -> float:
    losses = []
    for plc, plr, rlc, rlr in zip(pi_logp_chosen, pi_logp_rejected, ref_logp_chosen, ref_logp_rejected):
        margin = (plc - plr) - (rlc - rlr)
        losses.append(-math.log(sigmoid(beta * margin) + 1e-12))
    return sum(losses) / len(losses)


def demo() -> None:
    pi_logp_chosen = [-1.1, -0.9, -1.3]
    pi_logp_rejected = [-1.6, -1.2, -1.8]
    ref_logp_chosen = [-1.2, -1.0, -1.4]
    ref_logp_rejected = [-1.5, -1.1, -1.6]

    loss = dpo_loss(
        pi_logp_chosen=pi_logp_chosen,
        pi_logp_rejected=pi_logp_rejected,
        ref_logp_chosen=ref_logp_chosen,
        ref_logp_rejected=ref_logp_rejected,
        beta=0.1,
    )
    print(f"DPO loss={loss:.6f}")
    print("复杂度: O(n)")


if __name__ == "__main__":
    demo()

