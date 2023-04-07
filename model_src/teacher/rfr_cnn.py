import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DampConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, mt, mf):
        super(DampConv2d, self).__init__()
        self.mt = mt
        self.mf = mf
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding)

    def forward(self, x):  # 输入 (batch_size, channels, F, T)
        Fre_dims = x.shape[2]
        Time_dims = x.shape[3]
        C = torch.zeros(size=(Fre_dims, Time_dims)).to(device)  # C与输入维度一致
        for f in range(Fre_dims):
            for t in range(Time_dims):
                term1 = 1 - self.mt * np.abs(t - Time_dims / 2) / (Time_dims / 2)
                term2 = 1 - self.mf * np.abs(t - Fre_dims / 2) / (Fre_dims / 2)
                C[f][t] = term1 * term2
        C = torch.unsqueeze(C, dim=0)
        C = torch.unsqueeze(C, dim=0)
        x = torch.mul(x, C)
        x = self.conv2d(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), shortcut=False, dropout=False):
        super(BasicBlock, self).__init__()
        self.damp_conv1 = DampConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=1, padding='same', mt=0.9, mf=0.9)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.damp_conv2 = DampConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=1, padding='same', mt=0.9, mf=0.9)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.shortcut = shortcut
        self.dropout = dropout
        if shortcut:
            damp_convs_kernel_1 = kernel_size[0]
            damp_convs_kernel_2 = kernel_size[1]
            if not kernel_size[0] == 1:
                damp_convs_kernel_1 = kernel_size[0] + 2
            if not kernel_size[1] == 1:
                damp_convs_kernel_2 = kernel_size[1] + 2
            self.damp_convs = DampConv2d(in_channels=in_channels, out_channels=out_channels, stride=1, padding='same',
                                         kernel_size=(damp_convs_kernel_1, damp_convs_kernel_2), mt=0.9, mf=0.9)
            self.bn_s = nn.BatchNorm2d(num_features=out_channels)
        if dropout:
            self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        input = x
        x = self.damp_conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.damp_conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        if self.shortcut:
            redusial = self.bn_s(self.damp_convs(input))
            x = x + redusial
        if self.dropout:
            x = self.dropout(x)
        return x


class RFR_CNN(nn.Module):
    def __init__(self, in_channels=1, num_class=10, rho=7):  # input (1, 256, 64)
        super(RFR_CNN, self).__init__()

        # stage1
        self.damp_conv1 = DampConv2d(in_channels=in_channels, out_channels=128, kernel_size=5,
                                     stride=2, padding='valid', mt=0.9, mf=0.9)  # (128, 126, 30)
        self.bn1 = nn.BatchNorm2d(128)

        # stage2
        self.block21 = BasicBlock(128, 128, (3, 1))  # (128, 63, 15)
        self.maxpool = nn.MaxPool2d(2)  # (128, 63, 15)
        self.block22 = BasicBlock(128, 128, (3, 3))
        self.block23 = BasicBlock(128, 128, (3, 3))
        self.block24 = BasicBlock(128, 128, (3, 3))

        # stage3
        self.block31 = BasicBlock(128, 256, (3, 1), shortcut=True, dropout=True)  # (256, 63, 15)
        self.block32 = BasicBlock(256, 256, (1, 1))
        self.block33 = BasicBlock(256, 256, (1, 1))
        self.block34 = BasicBlock(256, 256, (1, 1))

        # stage4
        self.block41 = BasicBlock(256, 512, (1, 1), shortcut=True, dropout=True)  # (512, 63, 15)
        self.block42 = BasicBlock(512, 512, (1, 1))
        self.block43 = BasicBlock(512, 512, (1, 1))
        self.block44 = BasicBlock(512, 512, (1, 1))

        # stage5
        self.damp_conv2 = DampConv2d(in_channels=512, out_channels=num_class, kernel_size=1, padding='valid', stride=1,
                                     mt=0.9,
                                     mf=0.9)  # (10, 63, 15)
        self.bn2 = nn.BatchNorm2d(num_class)
        self.adavg = nn.AdaptiveAvgPool2d(1)  # (1, 10, 1, 1)

    def forward(self, x):
        x = self.damp_conv1(x)
        x = self.bn1(x)
        x = self.block24(self.block23(self.block22(self.maxpool(self.block21(x)))))
        x = self.block34(self.block33(self.block32((self.block31(x)))))
        x = self.block44(self.block43(self.block42((self.block41(x)))))
        x = self.adavg(self.bn2(self.damp_conv2(x)))
        x = F.softmax(x, dim=1)
        x = x.view(-1, x.shape[1])
        return x
