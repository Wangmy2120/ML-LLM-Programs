# 主题 10：K-Means 聚类
# 记忆点：重复两步直到收敛：1) 按最近中心分配簇 2) 更新每个簇中心。

from typing import List, Tuple

Point = List[float]


def squared_distance(a: Point, b: Point) -> float:
    """计算两点欧氏距离的平方。"""
    return sum((x - y) ** 2 for x, y in zip(a, b))


def assign_clusters(points: List[Point], centroids: List[Point]) -> List[int]:
    """分配步骤：每个样本归到最近的中心点。"""
    labels = []
    for point in points:
        best_idx = min(range(len(centroids)), key=lambda i: squared_distance(point, centroids[i]))
        labels.append(best_idx)
    return labels


def update_centroids(points: List[Point], labels: List[int], k: int) -> List[Point]:
    """更新步骤：按簇内样本均值更新中心点。"""
    dim = len(points[0])
    new_centroids = [[0.0] * dim for _ in range(k)]
    counts = [0] * k

    for point, label in zip(points, labels):
        counts[label] += 1
        for i in range(dim):
            new_centroids[label][i] += point[i]

    for c in range(k):
        if counts[c] == 0:
            continue
        new_centroids[c] = [value / counts[c] for value in new_centroids[c]]

    return new_centroids


def kmeans(points: List[Point], k: int, max_iter: int = 20) -> Tuple[List[Point], List[int]]:
    """最小 K-Means 迭代实现。"""
    centroids = [points[i][:] for i in range(k)]  # 固定初始化，便于复现
    labels = [0] * len(points)

    for _ in range(max_iter):
        new_labels = assign_clusters(points, centroids)
        # 标签不再变化时认为收敛。
        if new_labels == labels:
            break
        labels = new_labels
        centroids = update_centroids(points, labels, k)

    return centroids, labels


def demo() -> None:
    points = [[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0], [4.5, 5.0]]
    centroids, labels = kmeans(points, k=2)
    print("centroids:", [[round(v, 4) for v in c] for c in centroids])
    print("labels:", labels)
    print("复杂度: 每轮 O(n * k * d)，总计 O(max_iter * n * k * d)")


if __name__ == "__main__":
    demo()

