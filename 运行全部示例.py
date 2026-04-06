# 统一运行入口：依次演示 llm_ACM 中的核心手撕题

import importlib
from typing import List, Tuple


DemoConfig = List[Tuple[str, str]]


def run_demo(title: str, module_name: str) -> None:
    module = importlib.import_module(module_name)
    if not hasattr(module, "demo"):
        raise AttributeError(f"模块 {module_name} 缺少 demo()")
    print("=" * 18, title, "=" * 18)
    module.demo()
    print()


def main() -> None:
    demos: DemoConfig = [
        ("01 Softmax 与 Mask", "主题01_Softmax与掩码"),
        ("02 缩放点积注意力", "主题02_缩放点积注意力"),
        ("03 多头注意力", "主题03_多头注意力"),
        ("04 位置编码", "主题04_位置编码"),
        ("05 前馈网络", "主题05_前馈网络"),
        ("06 LayerNorm 与残差", "主题06_LayerNorm与残差连接"),
        ("07 Transformer 块", "主题07_Transformer块"),
        ("08 KV Cache", "主题08_KV缓存"),
        ("09 线性回归", "主题09_线性回归"),
        ("10 K-Means", "主题10_KMeans"),
        ("11 MHA / GQA", "主题11_MHA与GQA"),
        ("12 LoRA", "主题12_LoRA"),
        ("13 LN / RMSNorm", "主题13_LN与RMSNorm"),
        ("14 熵 / KL / 交叉熵", "主题14_熵KL交叉熵"),
        ("15 PPO", "主题15_PPO"),
        ("16 DPO", "主题16_DPO"),
        ("17 单层 Transformer", "主题17_单层Transformer"),
    ]

    for title, module_name in demos:
        run_demo(title, module_name)


if __name__ == "__main__":
    main()

