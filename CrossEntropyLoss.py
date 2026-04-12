import torch
import torch.nn.functional as F

# 假设我们有 2 个样本 (batch_size=2)，做 3 分类任务
# 这是模型最后一层输出的原始打分 (Logits)，注意：没有经过 Softmax!
logits = torch.tensor([[2.0, 1.0, 0.1],   # 样本 0 的打分
                       [0.5, 2.5, 0.3]])  # 样本 1 的打分

# 真实的标签：样本 0 是第 0 类，样本 1 是第 1 类
target = torch.tensor([0, 1])

# 第一步：Softmax，把无边界的打分变成 0-1 之间的概率分布
probs = F.softmax(logits, dim=-1)

# 第二步：取自然对数 Log
log_probs = torch.log(probs)

# 第三步：NLLLoss (负对数似然) - 只挑出正确类别的 log 概率，加个负号求平均
# target[0]=0，挑出 log_probs[0, 0]
# target[1]=1，挑出 log_probs[1, 1]
# 在 Python 里可以用高级索引一步写完：
correct_log_probs = log_probs[range(len(target)), target]
loss_manual = -correct_log_probs.mean()

print(f"手动拆解算出的 Loss: {loss_manual.item():.4f}")
