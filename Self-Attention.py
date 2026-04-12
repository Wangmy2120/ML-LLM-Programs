import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        # 定义生成 Q, K, V 的三个线性变换矩阵
        # 相当于公式里的 W_q, W_k, W_v
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # 输入 x 的形状通常是：(batch_size, seq_len, embed_dim)
        # 第一步：通过线性层，生成 Q, K, V
        Q = self.W_q(x)  # 形状：(batch_size, seq_len, embed_dim)
        K = self.W_k(x)  # 形状：(batch_size, seq_len, embed_dim)
        V = self.W_v(x)  # 形状：(batch_size, seq_len, embed_dim)
        
        # 第二步：计算注意力原始分数 (Q 乘以 K 的转置)
        # 注意：只能转置最后两个维度 (seq_len 和 embed_dim)，不能动 batch 维度
        K_transposed = K.transpose(-2, -1)
        scores = torch.matmul(Q, K_transposed)  # 形状：(batch_size, seq_len, seq_len)
        
        # 第三步：缩放（除以根号下 d_k）
        d_k = self.embed_dim
        scores = scores / math.sqrt(d_k)
        
        # 第四步：计算 Softmax，得到注意力权重
        # 在最后一个维度 (seq_len) 上进行归一化，保证每个词对其他词的注意力之和为 1
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 第五步：将权重作用于 V，得到最终输出
        output = torch.matmul(attn_weights, V)  # 形状：(batch_size, seq_len, embed_dim)
        
        return output
