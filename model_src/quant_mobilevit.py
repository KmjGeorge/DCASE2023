import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.jit.quantized import QuantizedLinear
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

    def fuse_model(self):
        fuse_modules(self.net, [['0', '1']], inplace=True)

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
        shape = qkv[0].shape
        b = shape[0]
        p = shape[1]
        n = shape[2]
        h = self.heads
        d = shape[3] // h
        q = qkv[0].reshape(b, p, h, n, d)
        k = qkv[1].reshape(b, p, h, n, d)
        v = qkv[2].reshape(b, p, h, n, d)
        # q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.reshape(b, p, n, h * d)
        return self.to_out(out)


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


class MV2Block(nn.Module):
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
        res1 = x

        y1 = self.depth_conv(x)
        y1 = self.point_conv(y1)

        res2 = y1

        _, _, h, w = x.shape
        patches, info_dict = self.unfolding(y1)
        # x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)

        patches = self.transformer(patches)
        # x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
        #              pw=self.pw)
        y2 = self.folding(patches=patches, info_dict=info_dict)

        y2 = self.conv3(y2)
        y2 = self.floatfunc.cat([y2, res2], 1)
        # y2 = torch.cat([y2, res2], dim=1)
        y2 = self.conv4(y2)
        y2 = self.floatfunc.add(y2, res1)
        # y2 = y2 + res1
        return y2

    # def forward(self, x):
    #     y1 = x.clone()
    #
    #     x = self.point_conv(self.depth_conv(x))
    #     y2 = x.clone()
    #
    #     _, _, h, w = x.shape
    #     x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
    #     x = self.transformer(x)
    #     x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
    #                   pw=self.pw)
    #
    #     x = self.conv3(x)
    #     x = self.floatfunc.cat([x, y2], 1)
    #     x = self.conv4(x)
    #     x = self.floatfunc.add(x, y1)
    #     return x


class MobileAST_Light(nn.Module):
    def __init__(self, spec_size, num_classes, expansion=2, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0

        self.conv1 = conv_nxn_bn(1, 32, stride=2)
        self.mv2_1 = MV2Block(32, 64, 1, expansion=1)
        self.mv2_2 = MV2Block(64, 64, 2, expansion=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.mvit_1 = MobileASTBlockv3(dim=32, heads=4, depth=2, channel=64, kernel_size=kernel_size,
                                       patch_size=patch_size, mlp_dim=64)
        self.mv2_3 = MV2Block(64, 64, 1, expansion=1)
        self.conv2 = conv_1x1_bn(64, 160)
        self.pool = nn.AvgPool2d((ih // 8, iw // 4), 1)
        self.fc = nn.Linear(160, num_classes, bias=False)

    def fuse_model(self):
        fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        self.mv2_1.fuse_model()
        self.mv2_2.fuse_model()
        self.mv2_3.fuse_model()
        self.mvit_1.fuse_model()
        fuse_modules(self.conv2, ['0', '1', '2'], inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.maxpool1(x)
        x = self.mvit_1(x)
        x = self.mv2_3(x)
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
        self.mvit_1 = MobileASTBlockv3(dim=32, heads=4, depth=2, channel=32, kernel_size=kernel_size,
                                       patch_size=patch_size, mlp_dim=64)
        self.mv2_3 = MV2Block(32, 64, 2, expansion=2)

        self.conv2 = conv_1x1_bn(64, 160)
        self.pool = nn.AvgPool2d((16, 8), 1)
        self.fc = nn.Linear(160, num_classes, bias=False)

    def fuse_model(self):
        fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        fuse_modules(self.conv2, ['0', '1', '2'], inplace=True)
        self.mv2_1.fuse_model()
        self.mv2_2.fuse_model()
        self.mv2_3.fuse_model()
        self.mvit_1.fuse_model()

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.fmaxpool(x)
        x = self.mvit_1(x)
        x = self.mv2_3(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, x.shape[1])
        x = self.fc(x)

        return x


class MobileAST_Light3(nn.Module):
    def __init__(self, spec_size, num_classes, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()
        self.conv1 = conv_nxn_bn(1, 32, stride=2)
        self.mv2_1 = MV2Block(32, 32, 1, expansion=1)
        self.mv2_2 = MV2Block(32, 32, 2, expansion=1)
        self.fmaxpool = nn.MaxPool2d(kernel_size=(2, 1))
        self.mvit_1 = MobileASTBlockv3(dim=48, heads=4, depth=2, channel=32, kernel_size=kernel_size,
                                       patch_size=patch_size, mlp_dim=64)
        self.mv2_3 = MV2Block(32, 64, 2, expansion=1)
        self.conv2 = conv_1x1_bn(64, 128)
        self.pool = nn.AvgPool2d((16, 8), 1)
        self.fc = nn.Linear(128, num_classes, bias=False)

    def fuse_model(self):
        fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        fuse_modules(self.conv2, ['0', '1', '2'], inplace=True)
        self.mv2_1.fuse_model()
        self.mv2_2.fuse_model()
        self.mv2_3.fuse_model()
        self.mvit_1.fuse_model()

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.fmaxpool(x)
        x = self.mvit_1(x)
        x = self.mv2_3(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, x.shape[1])
        x = self.fc(x)

        return x


class MobileAST_Light4(nn.Module):
    def __init__(self, spec_size, num_classes, kernel_size=(3, 3), patch_size=(2, 2),
                 ):
        super().__init__()

        ih, iw = spec_size[0], spec_size[1]
        ph, pw = patch_size[0], patch_size[1]
        assert ih % ph == 0 and iw % pw == 0

        self.conv1 = conv_nxn_bn(1, 32, stride=2, kernal_size=5)
        self.mv2_1 = MV2Block(32, 48, stride=2, expansion=1)
        self.fmaxpool = nn.MaxPool2d(kernel_size=(2, 1))
        self.mvit_1 = MobileASTBlockv3(dim=64, heads=4, depth=2, channel=48, kernel_size=kernel_size,
                                       patch_size=patch_size, mlp_dim=64)
        # self.conv2 = conv_1x1_bn(48, 96)
        self.pool = nn.AvgPool2d((32, 16), 1)
        self.fc = nn.Linear(48, num_classes, bias=False)

    def fuse_model(self):
        fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        fuse_modules(self.conv2, ['0', '1', '2'], inplace=True)
        self.mv2_1.fuse_model()
        self.mv2_2.fuse_model()
        self.mv2_3.fuse_model()
        self.mvit_1.fuse_model()

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.fmaxpool(x)
        x = self.mvit_1(x)
        # x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, x.shape[1])
        x = self.fc(x)

        return x


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


def mobileast_light3(mixstyle_conf):
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(p=mixstyle_conf['p'], alpha=mixstyle_conf['alpha'], freq=mixstyle_conf['freq']),
                             MobileAST_Light3((256, 64), num_classes=10, kernel_size=(3, 3),
                                              patch_size=(2, 2)).to(device))
    else:
        return MobileAST_Light3((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2)).to(device)


def mobileast_light4(mixstyle_conf):
    if mixstyle_conf['enable']:
        return nn.Sequential(MixStyle(p=mixstyle_conf['p'], alpha=mixstyle_conf['alpha'], freq=mixstyle_conf['freq']),
                             MobileAST_Light4((256, 64), num_classes=10, kernel_size=(3, 3),
                                              patch_size=(2, 2)).to(device))
    else:
        return MobileAST_Light4((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2)).to(device)


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
    print('Size (KB):', os.path.getsize("temp.p") / 1024)
    # print('Size (MB):', os.path.getsize("1234.p") / 1e6)
    # os.remove('temp.p')
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

    input_shape = (1, 1, 256, 64)
    # model_fp32 = Test((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2))

    model_fp32 = nn.Sequential(ExtractMel(**spectrum_config), MobileAST_Light4((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2)))
    # model_fp32 = MobileAST_Light4((256, 64), num_classes=10, kernel_size=(3, 3), patch_size=(2, 2))
    # torch.save(model_fp32.state_dict(), '../123.pt')
    # model_fp32 = mobileast_light2(mixstyle_config)
    # model_fp32 = MobileASTBlockv3(dim=32, heads=4, depth=2, channel=32, kernel_size=(3, 3),
    #                               patch_size=(2, 2), mlp_dim=32)
    # model_fp32 = Transformer(dim=32, depth=2, heads=4, mlp_dim=32)
    nessi.get_model_size(model_fp32, 'torch', input_size=(1, 32000))
    assert False
    # model_fp32 = cp_resnet(mixstyle_config, rho=8, cut_channels_s3=36, s2_group=2, n_blocks=(2, 1, 1))
    model_fp32.eval()
    print_size_of_model(model_fp32)
    input_fp32 = torch.rand(input_shape)
    '''
    # ptsq
    model_fp32.fuse_model()
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

    # tau2022_random_slicing_test = TAU2022('../dataset/h5/tau2022_test.h5', mel=True)
    # calibration_set = DataLoader(tau2022_random_slicing_test, batch_size=64, shuffle=False)
    # for i, (x, _, _) in enumerate(calibration_set):
    #     x = x.to('cpu')
    #     if i == 50:
    #         break
    #     model_fp32_prepared(x)
    # print('校准完成')

    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    print_size_of_model(model_int8)
    '''

    # fx
    model_to_quantize = copy.deepcopy(model_fp32)
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    example_inputs = (input_fp32)
    model_fp32_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    model_int8 = quantize_fx.convert_fx(model_fp32_prepared)
    print_size_of_model(model_int8)

    rlt = model_int8(input_fp32)
    print(rlt)
    '''
    for (k1, v1), (k2, v2) in zip(model_fp32.state_dict().items(), model_int8.state_dict().items()):
        try:
            print(k1, v1.type(), v2.type(), v1.type() == v2.type())
        except:
            pass
    '''

    nessi.get_model_size(model_fp32, 'torch', input_size=input_shape)
    nessi.get_model_size(model_fp32_prepared, 'torch', input_size=input_shape)
