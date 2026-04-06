# 主题 9：线性回归（Linear Regression）
# 记忆点：y_hat = w * x + b，损失用 MSE，梯度下降更新参数。

from typing import List, Tuple


def predict(x: List[float], w: float, b: float) -> List[float]:
    """线性模型前向：y_hat = w*x + b。"""
    return [w * xi + b for xi in x]


def mse(y_hat: List[float], y: List[float]) -> float:
    """均方误差损失。"""
    n = len(y)
    return sum((y_hat[i] - y[i]) ** 2 for i in range(n)) / n


def train_linear_regression(
    x: List[float],
    y: List[float],
    lr: float = 0.01,
    epochs: int = 2000,
) -> Tuple[float, float]:
    """用批量梯度下降训练一元线性回归参数 w、b。"""
    w, b = 0.0, 0.0
    n = len(x)
    for _ in range(epochs):
        y_hat = predict(x, w, b)
        # 分别对 w、b 求导并更新。
        grad_w = (2.0 / n) * sum((y_hat[i] - y[i]) * x[i] for i in range(n))
        grad_b = (2.0 / n) * sum(y_hat[i] - y[i] for i in range(n))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def demo() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [3.0, 5.0, 7.0, 9.0]  # 真实关系约为 y = 2x + 1
    w, b = train_linear_regression(x, y)
    y_hat = predict(x, w, b)
    print(f"w={w:.4f}, b={b:.4f}")
    print("预测:", [round(v, 4) for v in y_hat])
    print(f"MSE={mse(y_hat, y):.6f}")
    print("复杂度: 每轮 O(n)，总计 O(epochs * n)")


if __name__ == "__main__":
    demo()

