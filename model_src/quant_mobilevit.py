import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataset.datagenerator import TAU2022
from model_src.module.mixstyle import MixStyle
from model_src.module.ssn import SubSpecNormalization
from model_src.module.rfn import RFN
from torch.quantization import QuantStub, DeQuantStub
from model_src.module.grl import GRL
from torch.quantization import fuse_modules
from torch.nn.quantized import FloatFunctional
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 子谱数
SubSpecNum = 2

layer_index_total = 0


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def conv_1x1_ssn(inp, oup, S):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        SubSpecNormalization(oup, S=S),
        nn.ReLU()
    )


def conv_nxn_ssn(inp, oup, S, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        SubSpecNormalization(oup, S=S),
        nn.ReLU()
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

'''
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def fuse_model(self):
        fuse_modules(self.net, [['0', '1']], inplace=True)

    def forward(self, x):
        return self.net(x)


class FeedForward_conv(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def fuse_model(self):
        fuse_modules(self.net, [['0', '1']], inplace=True)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b w c h')
        x = self.net(x)
        x = rearrange(x, 'b w c h -> b c h w')
        return x

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

        kt = k.transpose(-1, -2)
        dots = torch.matmul(q, kt) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)
        
class Attention_Conv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        out = rearrange(out, 'b p n hd -> b hd p n')
        out = self.to_out(out)
        out = rearrange(out, 'b c h w -> b h w c')
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, ff_dropout=0., attn_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        dim_head = dim // heads
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(dim),
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, ff_dropout)
            ))
        self.floatfunc = FloatFunctional()

    def fuse_model(self):
        for layer in self.layers:
            layer[3].fuse_model()

    def forward(self, x):
        for layer in self.layers:
            x = layer[0](x)
            x = self.floatfunc.add(layer[1](x), x)
            x = layer[2](x)
            x = self.floatfunc.add(layer[3](x), x)
        return x
'''


class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, mlp_dim, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(dim),
            nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=attn_dropout, bias=False),
            nn.Dropout(p=ff_dropout),
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(p=ff_dropout),
            nn.Linear(mlp_dim, dim, bias=False),
            nn.Dropout(p=ff_dropout),
        )
        self.head = heads

        self.floatfunc = FloatFunctional()

    def fuse_model(self):
        fuse_modules(self.pre_norm_ffn, [['1', '2']], inplace=True)

    def forward(self, x):
        res = x.clone()
        shape = x.shape
        x = self.pre_norm_mha[0](x)  # norm
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        # x = rearrange(x, 'b head h w -> (b head) h w', head=self.head)
        x, _ = self.pre_norm_mha[1](x, x, x)  # mha
        x = x.view(shape)
        # x = rearrange(x, '(b head) h w -> b head h w', head=self.head)
        x = self.pre_norm_mha[2](x)  # dropout
        x = self.floatfunc.add(x, res)

        # Feed forward network
        x = self.floatfunc.add(x, self.pre_norm_ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.layers = nn.Sequential()
        for _ in range(depth):
            self.layers.append(TransformerEncoder(
                dim=dim,
                heads=heads,
                mlp_dim=mlp_dim,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout))
        self.norm = nn.LayerNorm(dim)

    def fuse_model(self):
        for layer in self.layers:
            layer.fuse_model()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class MV2Block(nn.Module):
    """
    :param inp 输入通道
    :param oup 输出通道
    :param stride 当stride=1，且输入通道=输出通道时才有shotcut
    :param expansion 扩展因子
    """

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expansion = expansion
        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.floatfunc = FloatFunctional()

    def fuse_model(self):
        if self.expansion == 1:
            fuse_modules(self.conv, [['0', '1', '2'], ['3', '4']], inplace=True)
        else:
            fuse_modules(self.conv, [['0', '1', '2'], ['3', '4', '5'], ['6', '7']], inplace=True)

    def forward(self, x):
        if self.use_res_connect:
            return self.floatfunc.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileASTBlockv3(nn.Module):
    def __init__(self, dim, heads, depth, channel, kernel_size, patch_size, mlp_dim, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size[0], patch_size[1]

        self.depth_conv = nn.Conv2d(channel, channel, kernel_size, groups=channel, padding=1)
        self.point_conv = nn.Conv2d(channel, dim, kernel_size=1)

        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, ff_dropout=ff_dropout,
                                       attn_dropout=attn_dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_1x1_bn(dim + channel, channel)
        self.floatfunc = FloatFunctional()

    def fuse_model(self):
        fuse_modules(self.conv3, ['0', '1', '2'], inplace=True)
        fuse_modules(self.conv4, ['0', '1', '2'], inplace=True)
        self.transformer.fuse_model()

    def folding(self, patches, info_dict):
        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(
            batch_size * channels * num_patch_h, num_patch_w, self.ph, self.pw
        )
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            batch_size, channels, num_patch_h * self.ph, num_patch_w * self.pw
        )

        return feature_map

    def unfolding(self, feature_map):
        patch_w, patch_h = self.pw, self.ph
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = math.ceil(orig_h / self.ph) * self.ph
        new_w = math.ceil(orig_w / self.pw) * self.pw

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(
            batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w
        )
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(
            batch_size, in_channels, num_patches, patch_area
        )
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size, patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def forward(self, x):
        y1 = x.clone()

        x = self.point_conv(self.depth_conv(x))
        y2 = x.clone()

        _, _, h, w = x.shape
        patches, info_dict = self.unfolding(x)
        # x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)

        patches = self.transformer(patches)
        # x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
        #              pw=self.pw)
        x = self.folding(patches=patches, info_dict=info_dict)

        x = self.conv3(x)
        x = self.floatfunc.cat([x, y2], 1)
        x = self.conv4(x)
        x = self.floatfunc.add(x, y1)
        return x

    '''
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
        x = self.floatfunc.cat([x, y2], 1)
        x = self.conv4(x)
        x = self.floatfunc.add(x, y1)
        return x
    '''


class MobileASTBlock(nn.Module):
    def __init__(self, dim, depth, heads, channel, kernel_size, patch_size, mlp_dim, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size[0], patch_size[1]

        self.conv1 = conv_nxn_bn(channel, channel, kernal_size=kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                       attn_dropout=attn_dropout, ff_dropout=ff_dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(channel * 2, channel, kernal_size=kernel_size)
        self.floatfunc = FloatFunctional()

    def fuse_model(self):
        fuse_modules(self.conv3, ['0', '1', '2'], inplace=True)
        fuse_modules(self.conv4, ['0', '1', '2'], inplace=True)

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
        x = self.floatfunc.cat([x, y], 1)  # (batch_size, d*2, h, w)
        x = self.conv4(x)  # (batch_size, d, h, w)

        return x


class MobileAST_Light(nn.Module):
    def __init__(self, spec_size, num_classes, expansion=2, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv1 = conv_nxn_bn(1, 32, stride=2)

        self.mv2_1 = MV2Block(32, 48, 1, expansion=1)
        self.mv2_2 = MV2Block(48, 48, 2, expansion=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1))
        # self.mv2_3 = MV2Block_SSN(32, 48, 1, expansion=1)
        # self.mv2_4 = MV2Block_SSN(48, 48, 2, expansion=1)
        self.mvit_1 = MobileASTBlockv3(dim=48, heads=4, depth=6, channel=48, kernel_size=kernel_size,
                                       patch_size=patch_size, mlp_dim=64)

        self.conv2 = conv_1x1_bn(48, 300)

        self.pool = nn.AvgPool2d((ih // 8, iw // 4), 1)
        self.fc = nn.Linear(300, num_classes, bias=False)

    def fuse_model(self):
        fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        self.mv2_1.fuse_model()
        self.mv2_2.fuse_model()
        self.mvit_1.fuse_model()
        fuse_modules(self.conv2, ['0', '1', '2'], inplace=True)

    def forward(self, x):
        x = self.quant(x)
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
        x = self.dequant(x)
        return x


class MobileAST_Light2_Quant(nn.Module):
    def __init__(self, spec_size, num_classes, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv1 = conv_nxn_bn(1, 32, stride=2)

        self.mv2_1 = MV2Block(32, 32, 1, expansion=2)
        self.mv2_2 = MV2Block(32, 32, 2, expansion=1)
        self.fmaxpool = nn.MaxPool2d(kernel_size=(2, 1))
        self.mvit_1 = MobileASTBlockv3(dim=32, heads=4, depth=2, channel=32, kernel_size=kernel_size,
                                       patch_size=patch_size, mlp_dim=32)
        self.mv2_3 = MV2Block(32, 64, 2, expansion=2)
        self.mvit_2 = MobileASTBlockv3(dim=32, heads=4, depth=4, channel=64, kernel_size=kernel_size,
                                       patch_size=patch_size, mlp_dim=32)
        self.conv2 = conv_1x1_bn(64, 96)
        # self.conv2 = conv_1x1_bn(64, 200)

        # self.pool = nn.AvgPool2d((16, 8), 1)
        # self.fc = nn.Linear(200, num_classes, bias=False)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(
                96,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def fuse_model(self):
        fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        self.mv2_1.fuse_model()
        self.mv2_2.fuse_model()
        self.mv2_3.fuse_model()
        self.mvit_1.fuse_model()
        self.mvit_2.fuse_model()
        fuse_modules(self.feed_forward, ['0', '1'], inplace=True)

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.fmaxpool(x)

        x = self.mvit_1(x)
        x = self.mv2_3(x)
        x = self.mvit_2(x)

        x = self.conv2(x)
        x = self.feed_forward(x)
        # x = self.pool(x)
        # x = x.view(-1, x.shape[1])
        # x = self.fc(x)

        x = self.dequant(x)
        x = x.squeeze(2).squeeze(2)
        return x


class MobileAST_Light2(nn.Module):
    def __init__(self, spec_size, num_classes, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv1 = conv_nxn_bn(1, 32, stride=2)

        self.mv2_1 = MV2Block(32, 32, 1, expansion=2)
        self.mv2_2 = MV2Block(32, 32, 2, expansion=1)
        self.fmaxpool = nn.MaxPool2d(kernel_size=(2, 1))
        self.mvit_1 = MobileASTBlockv3(dim=32, heads=4, depth=2, channel=32, kernel_size=kernel_size,
                                       patch_size=patch_size, mlp_dim=32)
        self.mv2_3 = MV2Block(32, 64, 2, expansion=2)
        self.mvit_2 = MobileASTBlockv3(dim=32, heads=4, depth=4, channel=64, kernel_size=kernel_size,
                                       patch_size=patch_size, mlp_dim=64)
        self.conv2 = conv_1x1_bn(64, 200)

        self.pool = nn.AvgPool2d((16, 8), 1)
        self.fc = nn.Linear(200, num_classes, bias=False)

        # self.feed_forward = nn.Sequential(
        #     nn.Conv2d(
        #         128,
        #         num_classes,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias=False),
        #     nn.BatchNorm2d(num_classes),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )

    def fuse_model(self):
        fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        self.mv2_1.fuse_model()
        self.mv2_2.fuse_model()
        self.mv2_3.fuse_model()
        self.mvit_1.fuse_model()
        self.mvit_2.fuse_model()
        fuse_modules(self.conv2, ['0', '1', '2'], inplace=True)

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.fmaxpool(x)

        x = self.mvit_1(x)
        x = self.mv2_3(x)
        x = self.mvit_2(x)

        x = self.conv2(x)
        # x = self.feed_forward(x)
        x = self.pool(x)
        x = x.view(-1, x.shape[1])
        x = self.fc(x)
        x = self.dequant(x)
        return x


class MobileAST_Light3(nn.Module):
    def __init__(self, spec_size, num_classes, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv1 = conv_nxn_bn(1, 32, stride=2)
        self.fmaxpool = nn.MaxPool2d(kernel_size=(2, 1))
        self.mvit_1 = MobileASTBlockv3(dim=32, heads=4, depth=2, channel=32, kernel_size=kernel_size,
                                        patch_size=patch_size, mlp_dim=64)
        self.conv2 = conv_nxn_bn(32, 64, stride=2)
        self.mvit_2 = MobileASTBlockv3(dim=48, heads=4, depth=4, channel=64, kernel_size=kernel_size,
                                        patch_size=patch_size, mlp_dim=64)
        self.conv3 = conv_1x1_bn(64, 128)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(
                128,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def fuse_model(self):
        fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        fuse_modules(self.conv2, ['0', '1', '2'], inplace=True)
        self.mv2_1.fuse_model()
        self.mv2_2.fuse_model()
        self.mv2_3.fuse_model()
        self.mvit_1.fuse_model()
        self.mvit_2.fuse_model()
        fuse_modules(self.feed_forward, ['0', '1'], inplace=True)

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.fmaxpool(x)

        x = self.mvit_1(x)
        x = self.conv2(x)
        x = self.mvit_2(x)

        x = self.conv3(x)
        x = self.feed_forward(x)
        # x = self.pool(x)
        # x = x.view(-1, x.shape[1])
        # x = self.fc(x)

        x = self.dequant(x)
        x = x.squeeze(2).squeeze(2)
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
        self.relu1 = nn.ReLU(True)
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
        self.relu2 = nn.ReLU(True)

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.skip_add.add(y, self.shortcut(x))
        y = self.relu2(y)
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
            nn.ReLU(True)
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.stage1 = _make_stage(in_channels=32, out_channels=32, n_blocks=1, block=CPBlock, maxpool=[0, 1],
                                  k1s=[3, 1], k2s=[1, 1], groups=1)

        self.mvit_1 = MobileASTBlock(dim=32, depth=8, heads=4, channel=32, kernel_size=kernel_size,
                                     patch_size=patch_size,
                                     mlp_dim=64)

        self.stage3 = _make_stage(in_channels=32, out_channels=64, n_blocks=1, block=CPBlock, maxpool=[], k1s=[1, 1],
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

    def fuse_model(self):
        fuse_modules(self.input_c, ['0', '1', '2'], inplace=True)
        fuse_modules(self.stage1[1], [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
        self.mvit_1.fuse_model()
        fuse_modules(self.stage3[0], [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
        fuse_modules(self.feed_forward, ['0', '1'], inplace=True)

    def forward(self, x):  # input (1, 1, 128, 64)
        x = self.quant(x)
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
        x = self.dequant(x)
        x = x.squeeze(2).squeeze(2)
        return x


class MobileAST_Light_CPResNet2(nn.Module):
    def __init__(self, spec_size, num_classes, kernel_size=(3, 3), patch_size=(2, 2), da_train=False):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.input_c = nn.Sequential(nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.stage1 = _make_stage(in_channels=32, out_channels=32, n_blocks=1, block=CPBlock, k1s=[3, 1], k2s=[3, 1],
                                  maxpool=[0, 1], groups=1)

        self.mvit_1 = MobileASTBlockv3(dim=32, depth=2, heads=4, channel=32, kernel_size=kernel_size,
                                       patch_size=patch_size,
                                       mlp_dim=64)

        self.mvit_2 = MobileASTBlockv3(dim=32, depth=2, heads=4, channel=32, kernel_size=kernel_size,
                                       patch_size=patch_size,
                                       mlp_dim=64)

        self.conv = conv_nxn_bn(32, 64, kernal_size=3, stride=2)

        self.stage2 = _make_stage(in_channels=64, out_channels=64, n_blocks=1, block=CPBlock, maxpool=[],
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

    def fuse_model(self):
        fuse_modules(self.input_c, ['0', '1', '2'], inplace=True)
        fuse_modules(self.stage1[1], [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
        self.mvit_1.fuse_model()
        fuse_modules(self.conv, ['0', '1', '2'], inplace=True)
        self.mvit_2.fuse_model()
        fuse_modules(self.stage2[0], [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
        fuse_modules(self.feed_forward, ['0', '1'], inplace=True)

    def forward(self, x):  # input (1, 1, 128, 64)
        x = self.quant(x)
        x = self.input_c(x)
        x = self.stage1(x)
        x = self.mvit_1(x)

        x = self.mvit_2(x)
        x = self.conv(x)
        x = self.stage2(x)
        # print(x.shape)
        x = self.feed_forward(x)
        x = self.dequant(x)
        x_class = x.squeeze(2).squeeze(2)
        if self.da_train:
            x_device = self.feed_forward_device(x).squeeze(2).squeeze(2)
            return x_class, x_device
        else:
            return x_class


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


def print_size_of_model(model):
    import os
    import pickle
    import gzip
    weights = []
    for k, v in model.state_dict().items():
        weights.append(v)
    # with gzip.open("1234.p", 'wb') as f:
    #     pickle.dump(weights, f)
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    # print('Size (MB):', os.path.getsize("1234.p") / 1e6)
    os.remove('temp.p')
    # os.remove("1234.p")


if __name__ == '__main__':
    from size_cal import nessi
    from configs.mixstyle import mixstyle_config
    from configs.dataconfig import spectrum_config
    from dataset.spectrum import ExtractMel
    from tqdm import tqdm
    from model_src.cp_resnet import cp_resnet
    from torch.ao.quantization import (
        get_default_qconfig_mapping,
        get_default_qat_qconfig_mapping,
        QConfigMapping,
    )
    import torch.ao.quantization.quantize_fx as quantize_fx
    import copy
    from model_src.cp_resnet import cp_resnet

    model_fp32 = MobileAST_Light2((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2))
    # model_fp32 = MobileASTBlockv3(dim=32, heads=4, depth=2, channel=32, kernel_size=(3, 3),
    #                                patch_size=(2, 2), mlp_dim=32)
    # model_fp32 = Transformer(dim=32, depth=2, heads=4, mlp_dim=32)
    # nessi.get_model_size(model_fp32, 'torch', input_size=(1, 1, 256, 64))
    # model_fp32 = cp_resnet(mixstyle_config, rho=8, cut_channels_s3=36, s2_group=2, n_blocks=(2, 1, 1))
    model_fp32.eval()
    print_size_of_model(model_fp32)

    # ptsq
    '''
    model_fp32.fuse_model()
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
    

    # tau2022_random_slicing_test = TAU2022('../dataset/h5/tau2022_test.h5', mel=True)
    # calibration_set = DataLoader(tau2022_random_slicing_test, batch_size=64, shuffle=False)
    # for i, (x, _, _) in enumerate(calibration_set):
    #     x = x.to('cpu')
    #     if i == 5:
    #         break
    #     model_fp32_prepared(x)
    # print('校准完成')

    model_int8_ptsq = torch.ao.quantization.convert(model_fp32_prepared)
    print_size_of_model(model_int8_ptsq)
    '''

    # fx
    model_to_quantize = copy.deepcopy(model_fp32)
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    input_fp32 = torch.rand(1, 1, 256, 64)
    example_inputs = (input_fp32)
    model_fp32_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    model_int8 = quantize_fx.convert_fx(model_fp32_prepared)
    print_size_of_model(model_int8)
    rlt = model_int8(input_fp32)
    print(rlt)

    for (k1, v1), (k2, v2) in zip(model_fp32.state_dict().items(), model_int8.state_dict().items()):
        try:
            print(k1, v1.type(), v2.type(), v1.type() == v2.type())
        except:
            pass

    nessi.get_model_size(model_fp32, 'torch', input_size=(1, 1, 256, 64))
    nessi.get_model_size(model_fp32_prepared, 'torch', input_size=(1, 1, 256, 64))

    # nessi.get_model_size(model_int8_ptsq, 'torch', input_size=(1, 1, 256, 64))
