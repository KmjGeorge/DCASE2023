import torch.nn as nn


# 子谱归一化
class SubSpecNormalization(nn.Module):
    """
    C: 输入通道数
    S: 子谱组数
    """
    def __init__(self, C, S):
        super(SubSpecNormalization, self).__init__()
        self.S = S
        self.bn = nn.BatchNorm2d(C * S)

    def forward(self, x):
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)
        x = self.bn(x)
        return x.view(N, C, F, T)
