import torch
import torch.nn as nn
from einops import rearrange
from model_src.module.mixstyle import MixStyle
from model_src.module.ssn import SubSpecNormalization
from model_src.module.rfn import RFN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 子谱数
SubSpecNum = 4


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_1x1_ssn(inp, oup, S):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        SubSpecNormalization(oup, S=S),
        nn.SiLU()
    )


def conv_nxn_ssn(inp, oup, S, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        SubSpecNormalization(oup, S=S),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_RFN(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = RFN(lamb=0.5, eps=1e-5)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer_Alter(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_RFN(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm_RFN(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    '''
    :param inp 输入通道
    :param oup 输出通道
    :param stride 当stride=1，且输入通道=输出通道时才有shotcut
    :param expansion 扩展因子
    '''

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MV2Block_SSN(nn.Module):
    '''
    :param inp 输入通道
    :param oup 输出通道
    :param stride 当stride=1，且输入通道=输出通道时才有shotcut
    :param expansion 扩展因子
    '''

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                SubSpecNormalization(hidden_dim, S=SubSpecNum),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                SubSpecNormalization(oup, S=SubSpecNum),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                SubSpecNormalization(hidden_dim, S=SubSpecNum),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                SubSpecNormalization(hidden_dim, S=SubSpecNum),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                SubSpecNormalization(oup, S=SubSpecNum),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_ssn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)

        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileASTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size[0], patch_size[1]

        self.conv1 = conv_nxn_ssn(channel, channel, S=SubSpecNum, kernal_size=kernel_size)
        self.conv2 = conv_1x1_ssn(channel, dim, S=SubSpecNum)

        self.transformer = Transformer_Alter(dim=dim, depth=depth, heads=4, dim_head=8, mlp_dim=mlp_dim,
                                             dropout=dropout)

        self.conv3 = conv_1x1_ssn(dim, channel, S=SubSpecNum)
        self.conv4 = conv_nxn_ssn(2 * channel, channel, S=SubSpecNum, kernal_size=kernel_size)

    def forward(self, x):
        y = x.clone()

        # CNN局部特征
        x = self.conv1(x)
        x = self.conv2(x)

        # Attention全局特征
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # 特征融合
        x = self.conv3(x)  # (batch_size, d, h, w)
        x = torch.cat((x, y), 1)  # (batch_size, d*2, h, w)
        x = self.conv4(x)  # (batch_size, d, h, w)

        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])

        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(
            MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(
            MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(
            MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)  # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])  # (b, 320, 1, 1) -> (b, 1, 320)
        x = self.fc(x)
        return x


# 将BN换为SSN
class MobileAST(nn.Module):
    def __init__(self, spec_size, dims, channels, num_classes, expansion=4, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()
        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        # dims = [64, 80, 96]
        # channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
        self.conv1 = conv_nxn_ssn(1, channels[0], stride=2, S=SubSpecNum)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block_SSN(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block_SSN(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block_SSN(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block_SSN(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block_SSN(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block_SSN(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block_SSN(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileASTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileASTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileASTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d((ih // 32, iw // 32), 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)

        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, x.shape[1])

        x = self.fc(x)
        return x


class MobileAST_Light(nn.Module):
    def __init__(self, spec_size, num_classes, expansion=2, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0

        self.conv1 = conv_nxn_ssn(1, 16, stride=2, S=SubSpecNum)

        self.mv2_1 = MV2Block_SSN(16, 24, 1, expansion)
        self.mv2_2 = MV2Block_SSN(24, 32, 2, expansion)

        self.mvit_1 = MobileASTBlock(48, 4, 32, kernel_size, patch_size, 96)

        self.mv2_3 = MV2Block_SSN(32, 64, 2, expansion)

        self.conv2 = conv_1x1_bn(64, 160)

        self.pool = nn.AvgPool2d((ih // 8, iw // 8), 1)
        self.fc = nn.Linear(160, num_classes, bias=False)

    def forward(self, x):  # input (1, 1, 128, 64)
        x = self.conv1(x)  # (1, 16, 64, 32)

        x = self.mv2_1(x)  # (1, 24, 64, 32)

        x = self.mv2_2(x)  # (1, 32, 32, 16)

        x = self.mvit_1(x)  # （1, 48, 32, 16)

        x = self.mv2_3(x)  # (1, 64, 16, 8)

        x = self.conv2(x)  # (1, 160, 16, 8)

        x = self.pool(x)  # (1, 160, 1, 1)

        x = x.view(-1, x.shape[1])  # (1, 160)

        x = self.fc(x)  # (1, 10)

        return x


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobileast_xxs(mixstyle_conf):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    # 频谱参数改变导致输入维度改变的时候，这里(128, 64)也要随之改变
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(mixstyle_conf['p'], mixstyle_conf['alpha'], mixstyle_conf['freq']),
                             MobileAST((128, 64), dims, channels,
                                       num_classes=10,
                                       expansion=2, kernel_size=(3, 3), patch_size=(2, 2)))
    else:
        return MobileAST((128, 64), dims, channels, num_classes=10, expansion=2, kernel_size=(3, 3),
                         patch_size=(2, 2))  # h=w<=n


def mobileast_light(mixstyle_conf):
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(mixstyle_conf['p'], mixstyle_conf['alpha'], mixstyle_conf['freq']),
                             MobileAST_Light((128, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2)).to(
                                 device))
    else:
        return MobileAST_Light((128, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2)).to(device)


def mobileast_s(mixstyle_conf):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    from model_src.module.mixstyle import MixStyle
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(mixstyle_conf['p'], mixstyle_conf['alpha'], mixstyle_conf['freq']),
                             MobileAST((128, 64), dims, channels, num_classes=10, kernel_size=(3, 3),
                                       patch_size=(2, 2)))

    else:
        return MobileAST((128, 64), dims, channels, num_classes=10, kernel_size=(3, 3), patch_size=(2, 2))


if __name__ == '__main__':
    # img = torch.randn(5, 3, 256, 256).to('cuda')

    from configs.mixstyle import mixstyle_config
    model = mobileast_light(mixstyle_conf=mixstyle_config)
    # model = mobileast_xxs(mixstyle_conf=mixstyle_config)
    # out = vit(img)
    # print(out.shape)
    # print(count_parameters(vit))

    # summary(vit, input_size=(3, 256, 256))

    # 评估模型参数量和MACC
    from size_cal import nessi

    nessi.get_model_size(model, 'torch', input_size=(1, 1, 128, 64))
    print(model)
