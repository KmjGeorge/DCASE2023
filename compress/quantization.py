import copy

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import get_default_qconfig_mapping


# 量化
def quantization(model_fp32, mode, calibration_set=None):
    model_fp32.cpu()
    model_fp32.eval()
    print('original:', end=' ')
    print_size_of_model(model_fp32)
    if mode == 'ptsq':
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model_fp32.fuse_model()
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
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
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        input_fp32 = torch.rand(1, 1, 256, 64)
        example_inputs = (input_fp32)
        model_fp32_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
        model_int8 = quantize_fx.convert_fx(model_fp32_prepared)
        print('int8:', end=' ')
        print_size_of_model(model_int8)
    else:
        raise 'mode error!'
    return model_int8


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size
