#softmax
import numpy as np

def softmax_final(x):
    # 确保是 numpy 数组
    x = np.array(x)
    
    # 如果是二维矩阵 (batch_size, num_classes)
    if x.ndim == 2:
        # 沿每一行取最大值，keepdims=True 保证形状对齐，方便减法
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shift_x)
        # 沿行求和归一化
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # 如果是一维向量
    else:
        shift_x = x - np.max(x)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x)

# 测试代码
data = np.array([[1, 2, 3], [1000, 1001, 1002]])
print(softmax_final(data))
