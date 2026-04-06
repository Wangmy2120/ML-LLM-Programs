"""
FreSHMoE.py
===========
简化版频域多专家系统（MoE）示例：
1) 按频率维分段，每段由一个小专家处理（Segment Experts）
2) 使用分段门控给各段加权融合（Segment Fusion Gate）
3) 再走全局专家并用全局门控聚合（Global Experts + Global Gate）
4) 用可学习系数融合局部与全局输出（alpha）

说明：
- 该文件用于提炼“如何实现多专家系统”的核心路径，尽量保留可读性。
- 仅新增独立文件，不接入 README 和“运行全部示例.py”。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreSHMoE(nn.Module):
    """
    简化版分段+全局 MoE：
    输入为复数频域张量 x_fft，形状约为 [B, 1, D]（兼容 [B, C, D]，但示例用 C=1）。
    """

    def __init__(
        self,
        segment_num: int = 4,
        experts_per_segment: int = 1,
        hidden: int = 32,
        global_experts_num: int = 2,
    ) -> None:
        super().__init__()
        self.segment_num = segment_num
        self.experts_per_segment = experts_per_segment
        self.hidden = hidden
        self.global_experts_num = global_experts_num

        self.initialized = False
        self.segment_len_padded = 0
        self.padded_len = 0

        # 懒初始化容器：依赖输入 D 后再构造。
        self.segment_experts = nn.ModuleList()
        # 懒初始化属性（依赖输入维度），首次 forward 时构造。
        self.global_experts = nn.ModuleList()
        self.segment_fusion_gate: Optional[nn.Sequential] = None
        self.global_gate: Optional[nn.Sequential] = None
        self.mix_gate: Optional[nn.Sequential] = None

        # 用 logit 参数化 alpha，前向用 sigmoid 映射到 [0, 1]。
        # 初始化时基础 alpha 为 0.5（随后还会被 dynamic_alpha 动态调制）。
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        # 便于调试/可视化的最近一次门控权重缓存。
        self.latest_gate_weights: Dict[str, torch.Tensor] = {}

    def _build_mlp(self, in_dim: int, out_dim: int) -> nn.Sequential:
        """构造最小两层 MLP 专家。"""
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, out_dim),
        )

    def lazy_init(self, d_model: int, device: torch.device) -> None:
        """根据输入最后一维长度 d_model 进行一次性初始化。"""
        if self.initialized:
            return

        self.segment_len_padded = (d_model + self.segment_num - 1) // self.segment_num
        self.padded_len = self.segment_len_padded * self.segment_num

        # 1) 分段专家
        for _ in range(self.segment_num):
            experts = nn.ModuleList(
                [
                    self._build_mlp(self.segment_len_padded, self.segment_len_padded)
                    for _ in range(self.experts_per_segment)
                ]
            )
            self.segment_experts.append(experts)

        # 2) 全局专家
        self.global_experts = nn.ModuleList(
            [self._build_mlp(self.padded_len, self.padded_len) for _ in range(self.global_experts_num)]
        )

        # 3) 分段融合门控：输出每个 segment 的权重
        self.segment_fusion_gate = nn.Sequential(
            nn.Linear(self.padded_len, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.segment_num),
        )

        # 4) 全局专家门控：输出每个 global expert 的权重
        self.global_gate = nn.Sequential(
            nn.Linear(self.padded_len, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.global_experts_num),
        )

        # 5) 局部/全局混合门控：输出 [0,1] 的动态混合系数
        self.mix_gate = nn.Sequential(
            nn.Linear(self.padded_len, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid(),
        )

        self.to(device)
        self.initialized = True

    def get_gate_weights(self) -> Dict[str, torch.Tensor]:
        """返回最近一次前向过程中的门控权重。"""
        return self.latest_gate_weights

    def _pad_to_segment_size(self, x: torch.Tensor, d_model: int) -> torch.Tensor:
        """把最后一维补齐到 segment_num * segment_len_padded，便于均匀分段。"""
        if d_model < self.padded_len:
            x = F.pad(x, (0, self.padded_len - d_model))
        return x

    def forward(self, x_fft: torch.Tensor) -> torch.Tensor:
        """
        前向：
        - 仅对实部做专家增强（简化实现）
        - 虚部保持原值，仅做 padding/truncate 对齐
        """
        if not torch.is_complex(x_fft):
            raise TypeError("x_fft 必须是复数张量（complex tensor）")

        real = x_fft.real  # [B, C, D]
        imag = x_fft.imag  # [B, C, D]
        if real.dim() != 3:
            raise ValueError("x_fft 期望形状为 [B, C, D]")

        bsz, channels, d_model = real.shape
        self.lazy_init(d_model=d_model, device=x_fft.device)

        real_padded = self._pad_to_segment_size(real, d_model)
        # 虚部不走专家网络，但需要同样 padding，保证最后复数重组时维度对齐。
        imag_padded = self._pad_to_segment_size(imag, d_model)

        # 按通道取平均作为所有门控与专家的共享输入特征（简化处理，避免对每通道独立计算）。
        # 这样得到形状 [B, D_pad] 的单路特征，随后在 local_moe/global_moe 处扩展回 [B, C, D_pad]。
        real_feature = real_padded.mean(dim=1)  # [B, D_pad]

        # ---- A. 分段专家路径 ----
        real_segments = torch.chunk(real_feature, self.segment_num, dim=-1)  # segment_num * [B, seg_len]
        segment_outputs = []
        for seg_idx in range(self.segment_num):
            expert_outputs = [expert(real_segments[seg_idx]) for expert in self.segment_experts[seg_idx]]
            if len(expert_outputs) == 1:
                seg_out = expert_outputs[0]
            else:
                seg_out = torch.mean(torch.stack(expert_outputs, dim=1), dim=1)
            segment_outputs.append(seg_out)

        moe_real_flat = torch.cat(segment_outputs, dim=-1)  # [B, D_pad]

        segment_logits = self.segment_fusion_gate(real_feature)  # [B, segment_num]
        segment_weights = F.softmax(segment_logits, dim=-1)  # [B, segment_num]

        # 按分段权重重加权
        weighted_segments = []
        moe_real_segments = torch.chunk(moe_real_flat, self.segment_num, dim=-1)
        for seg_idx in range(self.segment_num):
            seg_w = segment_weights[:, seg_idx].unsqueeze(-1)  # [B,1]
            weighted_segments.append(moe_real_segments[seg_idx] * seg_w)
        local_moe = torch.cat(weighted_segments, dim=-1).unsqueeze(1).expand(-1, channels, -1)  # [B, C, D_pad]

        # ---- B. 全局专家路径 ----
        global_input = moe_real_flat  # [B, D_pad]
        global_outputs = [expert(global_input) for expert in self.global_experts]  # N * [B, D_pad]
        global_stack = torch.stack(global_outputs, dim=1)  # [B, N, D_pad]

        global_logits = self.global_gate(global_input)  # [B, N]
        global_weights = F.softmax(global_logits, dim=-1).unsqueeze(-1)  # [B, N, 1]
        global_moe = torch.sum(global_stack * global_weights, dim=1).unsqueeze(1).expand(-1, channels, -1)

        # ---- C. 局部 + 全局融合 ----
        # alpha_total = sigmoid(alpha_logit) * dynamic_gate
        dynamic_alpha = self.mix_gate(global_input).unsqueeze(-1)  # [B,1,1]
        alpha_total = torch.sigmoid(self.alpha_logit) * dynamic_alpha

        # 残差连接 + 局部专家输出 + 加权全局专家输出
        final_real = real_padded + local_moe + alpha_total * global_moe
        final_imag = imag_padded

        # 截断回原始 D
        final_real = final_real[:, :, :d_model]
        final_imag = final_imag[:, :, :d_model]

        # 缓存门控权重，方便外部查看
        self.latest_gate_weights = {
            "segment_weights": segment_weights.detach().cpu(),
            "global_weights": global_weights.squeeze(-1).detach().cpu(),
            "alpha_total": alpha_total.squeeze(-1).detach().cpu(),
        }

        return torch.complex(final_real, final_imag)


def demo() -> None:
    """最小运行示例。"""
    torch.manual_seed(0)

    # 构造 [B, C, D] 的复数频域输入
    bsz, channels, d_model = 2, 1, 10
    real = torch.randn(bsz, channels, d_model)
    imag = torch.randn(bsz, channels, d_model)
    x_fft = torch.complex(real, imag)

    model = FreSHMoE(
        segment_num=3,
        experts_per_segment=2,
        hidden=16,
        global_experts_num=2,
    )

    y_fft = model(x_fft)
    gates = model.get_gate_weights()

    print("输入形状:", tuple(x_fft.shape))
    print("输出形状:", tuple(y_fft.shape))
    print("segment gate (第1个样本):", [round(v, 4) for v in gates["segment_weights"][0].tolist()])
    print("global gate  (第1个样本):", [round(v, 4) for v in gates["global_weights"][0].tolist()])
    print("alpha_total  (第1个样本):", round(float(gates["alpha_total"][0, 0]), 4))


if __name__ == "__main__":
    demo()
