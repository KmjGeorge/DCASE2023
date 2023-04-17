import torch
import pytorch_quantization
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os


# 量化
# TODO
def quantization(model, mode):
    if mode == 'ptdq':
        quanted_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    elif model == 'ptq':
        model.eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model.fused = torch.ao.quantization.fuse_modules(model, [['conv', 'batchnorm']])
        # quanted_model =
    else:
        quanted_model = None
    return quanted_model


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


if __name__ == '__main__':
    '''
    a = torch.rand(size=(3, 3))
    print(a)
    a, scale = tensor_quant.tensor_quant(a, a.abs().max(), 8)
    print(a)
    print(scale)
    '''

    from model_src.mobilevit import mobileast_xxs
    from size_cal import nessi
    from torchsummary import summary
    from model_src.cp_resnet import cp_resnet
    from model_src.mobilevit import mobileast_light
    from configs.mixstyle import mixstyle_config

    # model = mobileast_xxs(mixstyle=True).to('cuda')
    model = mobileast_light(mixstyle_conf=mixstyle_config)
    qmodel = quantization(model, mode='ptdq')
    # summary(model, input_size=(1, 128, 64))
    # summary(qmodel, input_size=(1, 128, 64))
    nessi.get_model_size(model, 'torch', input_size=(1, 1, 128, 64))
    f = print_size_of_model(model, "fp32")
    q = print_size_of_model(qmodel, "int8")
    print("{0:.2f} times smaller".format(f / q))
