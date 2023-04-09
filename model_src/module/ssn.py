import torch.nn as nn


# 子谱归一化
class SubSpecNormalization(nn.Module):
    """
    gamma : 缩放参数
    beta : 偏移参数
    s: 子谱组数
    eps : 分母+eps防止为0
    """

    def __init__(self, gamma, beta, S, eps=1e-5):
        super(SubSpecNormalization, self).__init__()
        self.S = S
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    def forward(self, x):
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)
        mean = x.mean([0, 2, 3]).view(1, C * self.S, 1, 1)
        var = x.var([0, 2, 3]).view([1, C * self.S, 1, 1])
        x = self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta
        return x.view(N, C, F, T)
