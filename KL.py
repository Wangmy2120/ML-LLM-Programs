import torch
import torch.nn.functional as F

# 假设我们在做一个 3 分类任务
# P: 真实的概率分布 (比如教师模型给出的软标签)
P = torch.tensor([0.7, 0.2, 0.1])

# Q: 我们自己模型的预测概率分布 (已经做过 Softmax)
Q = torch.tensor([0.5, 0.4, 0.1])

# 为了防止取 log(0) 导致数值爆炸 (NaN)，通常加上一个极小的数 epsilon
eps = 1e-8

# =========================================
# 公式: Sum(P * log(P / Q))
# =========================================
# 利用对数性质 log(A/B) = log(A) - log(B)，这样写数值更稳定
kl_manual = torch.sum(P * (torch.log(P + eps) - torch.log(Q + eps)))
print(f"手撕版 KL 散度: {kl_manual.item():.4f}")

# ---------------------------------------------------------
# 3) Merge student outputs -> single KL loss
# ---------------------------------------------------------
with autocast():
    weights = self._compute_merge_weights(student_logits)
    
    if self.merge_target == "logits":
        # Weighted sum of logits -> then softmax for KL
        merged = sum(w * l for w, l in zip(weights, student_logits))
        logp_merged = F.log_softmax(merged, dim=-1)
        
    elif self.merge_target == "probability":
        # Weighted sum of probabilities -> then log for KL
        probs = [F.softmax(l, dim=-1) for l in student_logits]
        merged_prob = sum(w * p for w, p in zip(weights, probs))
        # Clamp to avoid log(0)
        logp_merged = torch.log(merged_prob.clamp_min(1e-12))
    
    # Per-sample KL: [B, C] -> [B]
    kl = F.kl_div(logp_merged, p_teacher, reduction="none").sum(dim=-1)
    kl = (kl * mask).sum() / denom
