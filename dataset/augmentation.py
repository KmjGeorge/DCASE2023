import torch
import numpy as np


# 对batch执行mixup，返回新的batch
# 注意mixup的实现还有一部分在训练阶段，见train.normal_train.train_per_epoch
def mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha)  # 采样λ
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)  # 随机选取混合下标
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam




