import copy

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import torch.ao.quantization.quantize_fx as quantize_fx
import tqdm

from torch.ao.quantization import get_default_qconfig_mapping

import model_src.cp_resnet
from size_cal import nessi


# 量化
def quantization(model, mode, calibration_set=None, show=True, backend='x86'):
    model.cpu()
    model.eval()
    print('original:', end=' ')
    model_fp32 = model[1][1]  # 取出主干网络  model[0]为mel提取器， 如果没有mixstyle model[1]为主干网络，否则为model[1][1]
    print_size_of_model(model_fp32)
    if mode == 'ptsq':
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        model_fp32.fuse_model()
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        if show:
            nessi.get_model_size(model_fp32, 'torch', input_size=(1, 1, 256, 64))

        if calibration_set:
            loop = tqdm.tqdm(calibration_set)
            for x in loop:
                x = x.to('cpu')
                x = model[0](x)
                model_fp32_prepared(x)
            print('校准完成')
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        print('int8:', end=' ')
        print_size_of_model(model_int8)
    elif mode == 'fx':
        model_to_quantize = copy.deepcopy(model_fp32)
        qconfig_mapping = get_default_qconfig_mapping(backend)
        input_fp32 = torch.rand(1, 1, 256, 64)
        example_inputs = (input_fp32)
        model_fp32_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
        if calibration_set:
            loop = tqdm.tqdm(calibration_set)
            for x in loop:
                x = x.to('cpu')
                x = model[0](x)
                model_fp32_prepared(x)
                loop.set_description('Calibration Epoch:')
            print('校准完成')
        model_int8 = quantize_fx.convert_fx(model_fp32_prepared)
        if show:
            nessi.get_model_size(model_fp32, 'torch', input_size=(1, 1, 256, 64))

        print('int8:', end=' ')
        print_size_of_model(model_int8)
    else:
        raise 'mode error!'

    model[1][1] = model_int8
    return model


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print('Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


if __name__ == '__main__':
    mixstyle_conf = {'enable': False}
    model = model_src.cp_resnet.cp_resnet(mixstyle_conf, rho=7, s2_group=2, cut_channels_s3=24, n_blocks=(2, 1, 1))
    model_int8 = quantization(model, 'ptsq', backend='x86')
