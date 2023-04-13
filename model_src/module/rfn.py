import torch.nn as nn


# 松弛实例频率归一化 Relaxed instance frequency-wise normalization
class RFN(nn.Module):
    """
    lamb: 松弛因子
    eps: 防止方差为0
    """
    def __init__(self, lamb, eps=1e-5):
        super(RFN, self).__init__()
        self.lamb = lamb
        self.eps = eps

    def forward(self, x):
        # x.shape = (N, C, F, T)
        # 频率上的实例归一化
        mean_ifn = x.mean(dim=(1, 3), keepdim=True)
        var_ifn = x.var(dim=(1, 3), keepdim=True, unbiased=False) + self.eps
        std_ifn = var_ifn.sqrt()

        # 层归一化
        mean_ln = x.mean(dim=(1, 2, 3), keepdim=True)
        var_ln = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False) + self.eps
        std_ln = var_ln.sqrt()

        # RFN
        x = self.lamb * (x-mean_ln) / std_ln + (1-self.lamb) * (x-mean_ifn) / std_ifn
        return x
