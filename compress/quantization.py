import copy

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import torch.ao.quantization.quantize_fx as quantize_fx

from torch.ao.quantization import get_default_qconfig_mapping

import model_src.cp_resnet
from size_cal import nessi


# 量化
def quantization(model_fp32, mode, calibration_set=None, show=True, backend='x86'):
    model_fp32.cpu()
    model_fp32.eval()
    print('original:', end=' ')
    print_size_of_model(model_fp32)

    if mode == 'ptsq':
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        model_fp32.fuse_model()
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        if show:
            nessi.get_model_size(model_fp32, 'torch', input_size=(1, 1, 256, 64))

        if calibration_set:
            for i, (x, _, _) in enumerate(calibration_set):
                x = x.to('cpu')
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
        if show:
            nessi.get_model_size(model_fp32, 'torch', input_size=(1, 1, 256, 64))

        model_int8 = quantize_fx.convert_fx(model_fp32_prepared)
        print('int8:', end=' ')
        print_size_of_model(model_int8)
    else:
        raise 'mode error!'
    return model_int8


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
