# 主题 15：PPO（Proximal Policy Optimization）
# 记忆点：
# 1) ratio = pi_new(a|s) / pi_old(a|s)
# 2) 用 clip 限制更新幅度，防止策略骤变
# 3) 最终目标：min(ratio*A, clip(ratio)*A)

from typing import List


def ppo_clipped_objective(
    old_probs: List[float],
    new_probs: List[float],
    advantages: List[float],
    eps: float = 0.2,
) -> float:
    terms = []
    for old_p, new_p, adv in zip(old_probs, new_probs, advantages):
        ratio = new_p / (old_p + 1e-12)
        clipped_ratio = min(max(ratio, 1.0 - eps), 1.0 + eps)
        terms.append(min(ratio * adv, clipped_ratio * adv))
    return sum(terms) / len(terms)


def demo() -> None:
    old_probs = [0.2, 0.5, 0.3]
    new_probs = [0.25, 0.45, 0.35]
    advantages = [1.0, -0.5, 0.8]

    objective = ppo_clipped_objective(old_probs, new_probs, advantages, eps=0.2)
    print(f"PPO clipped objective={objective:.6f}")
    print("复杂度: O(n)")


if __name__ == "__main__":
    demo()

