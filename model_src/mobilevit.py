import torch
import torch.nn as nn
from einops import rearrange
from model_src.module.mixstyle import MixStyle
from model_src.module.ssn import SubSpecNormalization
from model_src.module.rfn import RFN
from torch.quantization import QuantStub, DeQuantStub
from model_src.module.grl import GRL

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 子谱数
SubSpecNum = 2

layer_index_total = 0


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
            nn.SiLU(True),
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
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

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


class MobileASTBlockv3(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size[0], patch_size[1]

        self.depth_conv = nn.Conv2d(channel, channel, kernel_size, groups=channel, padding=1)
        self.point_conv = nn.Conv2d(channel, dim, kernel_size=1)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_1x1_bn(dim + channel, channel)

    def forward(self, x):
        y1 = x.clone()

        x = self.point_conv(self.depth_conv(x))
        y2 = x.clone()

        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        x = self.conv3(x)
        x = torch.cat((x, y2), 1)
        x = self.conv4(x)
        x = x + y1
        return x


class MobileASTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size[0], patch_size[1]

        self.conv1 = conv_nxn_bn(channel, channel, kernal_size=kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim=dim, depth=depth, heads=4, dim_head=8, mlp_dim=mlp_dim,
                                       dropout=dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(channel * 2, channel, kernal_size=kernel_size)

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

        self.conv1 = conv_nxn_ssn(1, 32, stride=2, S=SubSpecNum)

        self.mv2_1 = MV2Block(32, 48, 1, expansion=1)
        self.mv2_2 = MV2Block(48, 48, 2, expansion=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1))
        # self.mv2_3 = MV2Block_SSN(32, 48, 1, expansion=1)
        # self.mv2_4 = MV2Block_SSN(48, 48, 2, expansion=1)
        self.mvit_1 = MobileASTBlockv3(48, 6, 48, kernel_size, patch_size, 48 * 2)

        self.conv2 = conv_1x1_bn(48, 300)

        self.pool = nn.AvgPool2d((ih // 8, iw // 4), 1)
        self.fc = nn.Linear(300, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.maxpool1(x)
        # x = self.mv2_3(x)
        # x = self.mv2_4(x)
        x = self.mvit_1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, x.shape[1])
        x = self.fc(x)

        return x


class MobileAST_Light2(nn.Module):
    def __init__(self, spec_size, num_classes, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0

        self.conv1 = conv_nxn_bn(1, 32, stride=2)

        self.mv2_1 = MV2Block(32, 32, 1, expansion=2)
        self.mv2_2 = MV2Block(32, 32, 2, expansion=1)
        self.fmaxpool = nn.MaxPool2d(kernel_size=(2, 1))
        self.mvit_1 = MobileASTBlockv3(32, 2, 32, kernel_size, patch_size, 32 * 2)
        self.mv2_4 = MV2Block(32, 64, 2, expansion=2)
        self.mvit_2 = MobileASTBlockv3(48, 4, 64, kernel_size, patch_size, 48 * 2)

        self.conv2 = conv_1x1_bn(64, 200)

        self.pool = nn.AvgPool2d((16, 8), 1)
        self.fc = nn.Linear(200, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.fmaxpool(x)
        # x = self.maxpool1(x)
        # x = self.mv2_3(x)
        # x = self.mv2_4(x)
        x = self.mvit_1(x)
        # x = self.mv2_3(x)
        x = self.mv2_4(x)
        x = self.mvit_2(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, x.shape[1])
        x = self.fc(x)

        return x


def _make_stage(in_channels, out_channels, n_blocks, block, maxpool=None, k1s=(3, 3, 3, 3, 3, 3),
                k2s=(3, 3, 3, 3, 3, 3), groups=1, ):
    """
    @param in_channels: in channels to the stage
    @param out_channels: out channels of the stage
    @param n_blocks: number of blocks in the stage
    @param block: block type in the stage ( basic block )
    @param maxpool: location of the max pooling in the stage
    @param k1s: list of first convolution kernels in the stage for each block one value should exist
    @param k2s: list of second convolution kernels in the stage for each block one value should exist
    @param groups: groups of the block , if group of a stage more than one , all convolutions of the stage are grouped
    @return: an object containing several resnet blocks with arbitrary config
    """
    if maxpool is None:
        maxpool = set()
    stage = nn.Sequential()
    if 0 in maxpool:
        stage.add_module("maxpool{}_{}".format(0, 0), nn.MaxPool2d(2, 2))
    for index in range(n_blocks):
        stage.add_module('block{}'.format(index + 1),
                         block(in_channels,
                               out_channels,
                               k1=k1s[index], k2=k2s[index], groups=groups, ))

        in_channels = out_channels
        # if index + 1 in maxpool:
        for m_i, mp_pos in enumerate(maxpool):
            if index + 1 == mp_pos:
                stage.add_module("maxpool{}_{}".format(index + 1, m_i),
                                 nn.MaxPool2d((2, 1), stride=(2, 1)))
    return stage


def calc_padding(kernal):
    """
    @param kernal: kernel input
    @return: calculates padding required to get the same shape before entering the convolution
    """
    try:
        return kernal // 3
    except TypeError:
        return [k // 3 for k in kernal]


class CPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k1=3, k2=3, groups=1):
        """
        @param in_channels: input channels to the block
        @param out_channels: output channels of the block
        @param k1: kernel size of the first convolution in the block
        @param k2:kernel size of the second convolution in the block
        @param groups: grouping applied to the block default 1 : No Grouping
        """
        super(CPBlock, self).__init__()
        global layer_index_total

        self.layer_index = layer_index_total
        layer_index_total = layer_index_total + 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=k1,
            padding=calc_padding(k1),
            bias=False,
            groups=groups)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu1 = nn.SiLU(True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=k2,
            padding=calc_padding(k2),
            bias=False,
            groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # without skip connection

        self.skip_add = nn.quantized.FloatFunctional()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                    groups=groups))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN
        self.silu2 = nn.SiLU(True)

    def forward(self, x):
        y = self.silu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        # residual connection with addition compatible with quantization
        y = self.skip_add.add(y, self.shortcut(x))
        y = self.silu2(y)  # apply ReLU after addition
        return y


class MobileAST_Light_CPResNet(nn.Module):
    def __init__(self, spec_size, num_classes, expansion=2, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0

        self.input_c = nn.Sequential(nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(True)
        )

        self.stage1 = _make_stage(in_channels=32, out_channels=32, n_blocks=1, block=CPBlock, maxpool=[0, 1],
                                  k1s=[3, 1], k2s=[1, 1], groups=1)

        self.mvit_1 = MobileASTBlock(dim=32, depth=8, channel=32, kernel_size=kernel_size, patch_size=patch_size,
                                     mlp_dim=32 * 2)

        self.stage3 = _make_stage(in_channels=64, out_channels=64, n_blocks=1, block=CPBlock, maxpool=[], k1s=[1, 1],
                                  k2s=[1, 1], groups=2)

        self.feed_forward = nn.Sequential(
            nn.Conv2d(
                64,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):  # input (1, 1, 128, 64)
        # x = self.quant(x)
        x = self.input_c(x)  # (1, 32, 64, 32)
        # print(x.shape)
        x = self.stage1(x)
        # print(x.shape)
        # x = self.stage2(x)
        x = self.mvit_1(x)
        # print(x.shape)
        x = self.stage3(x)
        # print(x.shape)
        x = self.feed_forward(x)
        # print(x.shape)
        # x = self.dequant(x)
        x = x.squeeze(2).squeeze(2)
        return x


class MobileAST_Light_CPResNet2(nn.Module):
    def __init__(self, spec_size, num_classes, kernel_size=(3, 3), patch_size=(2, 2), da_train=False):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0

        self.input_c = nn.Sequential(nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(True)
        )

        self.stage1 = _make_stage(in_channels=32, out_channels=32, n_blocks=1, block=CPBlock, k1s=[3, 1], k2s=[3, 1],
                                  maxpool=[0, 1], groups=1)

        self.mvit_1 = MobileASTBlockv3(dim=32, depth=2, channel=32, kernel_size=kernel_size, patch_size=patch_size,
                                       mlp_dim=32 * 2)

        self.conv = conv_nxn_bn(32, 64, kernal_size=3, stride=2)

        self.mvit_2 = MobileASTBlockv3(dim=32, depth=4, channel=64, kernel_size=kernel_size, patch_size=patch_size,
                                       mlp_dim=32 * 2)

        self.stage3 = _make_stage(in_channels=64, out_channels=64, n_blocks=1, block=CPBlock, maxpool=[],
                                  k1s=[1, 1], k2s=[1, 1],
                                  groups=1)

        self.feed_forward = nn.Sequential(
            nn.Conv2d(
                64,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.da_train = da_train

        if self.da_train:
            self.feed_forward_device = nn.Sequential(
                GRL(alpha=1),
                nn.Conv2d(
                    64,
                    64,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.Conv2d(
                    64,
                    2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(2),
                nn.AdaptiveAvgPool2d((1, 1))
            )

    def forward(self, x):  # input (1, 1, 128, 64)
        # x = self.quant(x)
        x = self.input_c(x)
        x = self.stage1(x)
        x = self.mvit_1(x)
        x = self.conv(x)
        x = self.mvit_2(x)
        x = self.stage3(x)
        # print(x.shape)
        x_class = self.feed_forward(x).squeeze(2).squeeze(2)

        if self.da_train:
            x_device = self.feed_forward_device(x).squeeze(2).squeeze(2)
            return x_class, x_device
        else:
            return x_class


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
        return nn.Sequential(MixStyle(p=mixstyle_conf['p'], alpha=mixstyle_conf['alpha'], freq=mixstyle_conf['freq']),
                             MobileAST((128, 64), dims, channels,
                                       num_classes=10,
                                       expansion=2, kernel_size=(3, 3), patch_size=(2, 2)))
    else:
        return MobileAST((128, 64), dims, channels, num_classes=10, expansion=2, kernel_size=(3, 3),
                         patch_size=(2, 2))  # h=w<=n


def mobileast_light(mixstyle_conf):
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(p=mixstyle_conf['p'], alpha=mixstyle_conf['alpha'], freq=mixstyle_conf['freq']),
                             MobileAST_Light((256, 64), num_classes=10, kernel_size=(3, 3),
                                             patch_size=(2, 2)).to(device))
    else:
        return MobileAST_Light((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2)).to(device)


def mobileast_light2(mixstyle_conf):
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(p=mixstyle_conf['p'], alpha=mixstyle_conf['alpha'], freq=mixstyle_conf['freq']),
                             MobileAST_Light2((256, 64), num_classes=10, kernel_size=(3, 3),
                                              patch_size=(2, 2)).to(device))
    else:
        return MobileAST_Light2((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2)).to(device)


def mobileast_cpresnet(mixstyle_conf):
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(p=mixstyle_conf['p'], alpha=mixstyle_conf['alpha'], freq=mixstyle_conf['freq']),
                             MobileAST_Light_CPResNet((256, 64), num_classes=10, kernel_size=(3, 3),
                                                      patch_size=(2, 2)).to(device))
    else:
        return MobileAST_Light_CPResNet((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2)).to(device)


def mobileast_cpresnet2(mixstyle_conf):
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(p=mixstyle_conf['p'], alpha=mixstyle_conf['alpha'], freq=mixstyle_conf['freq']),
                             MobileAST_Light_CPResNet2((256, 64), num_classes=10, kernel_size=(3, 3),
                                                       patch_size=(2, 2)).to(device))
    else:
        return MobileAST_Light_CPResNet2((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2)).to(device)


def mobileast_s(mixstyle_conf):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    from model_src.module.mixstyle import MixStyle
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(p=mixstyle_conf['p'], alpha=mixstyle_conf['alpha'], freq=mixstyle_conf['freq']),
                             MobileAST((128, 64), dims, channels, num_classes=10, kernel_size=(3, 3),
                                       patch_size=(2, 2)))

    else:
        return MobileAST((128, 64), dims, channels, num_classes=10, kernel_size=(3, 3), patch_size=(2, 2))


if __name__ == '__main__':
    from size_cal import nessi
    from configs.mixstyle import mixstyle_config

    mobileast_light = mobileast_light(mixstyle_config)
    # mobileast_light2 = MobileAST_Light2(spec_size=(256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2))

    # nessi.get_model_size(blockv1, 'torch', (1, 32, 32, 16))
    # nessi.get_model_size(blockv3, 'torch', (1, 32, 32, 16))
    # model = mobileast_xxs(mixstyle_conf=mixstyle_config)
    # out = vit(img)
    # print(out.shape)
    # print(count_parameters(vit))

    # summary(vit, input_size=(3, 256, 256))

    # 评估模型参数量和MACC

    # nessi.get_model_size(model, 'torch', (1, 1, 256, 64))
    nessi.get_model_size(mobileast_light, 'torch', (1, 1, 256, 64))
    print(mobileast_light)
    # nessi.get_model_size(stage1, 'torch', (1, 24, 128, 32))
    # nessi.get_model_size(mobileastblock, 'torch', input_size=(1, 64, 64, 32))
    # nessi.get_model_size(model, 'torch', input_size=(1, 1, 128, 64))
