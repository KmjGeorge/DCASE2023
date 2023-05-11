import torch
import dataset.datagenerator
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import random
import numpy as np

from model_src.cp_resnet_freq_damp import get_model_based_on_rho
from model_src.passt import passt
from model_src.mobilevit import mobileast_light, mobileast_light2, mobileast_cpresnet2
from model_src.cp_resnet import cp_resnet
from configs.dataconfig import dataset_config, spectrum_config
from configs.trainingconfig import normal_training_config
from configs.mixup import mixup_config
from configs.mixstyle import mixstyle_config

from dataset.datagenerator import TAU2022
from torch.utils.data import DataLoader
from train.normal_train import validate
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # 固定种子
    setup_seed(200)
    from dataset.datagenerator import get_valset, get_calibration_set
    from compress.quantization import quantization
    from model_src.mobilevit import mobileast_light2
    import model_src.quant_mobilevit
    from dataset.spectrum import ExtractMel
    from size_cal import nessi

    extmel = ExtractMel(**spectrum_config)
    model = nn.Sequential(extmel, model_src.quant_mobilevit.mobileast_light(mixstyle_config)).to('cpu')
    # nessi.get_model_size(model, 'torch', (1, 32000))
    weight_path = '../model_weights/best/passt+mobileastv3_light1 nomvit2(fc) 1e-3 30 1e-4 400 DKD T=4 alpha=1 beta=10 MixStyle(0.3 0.6) Mixup(0.3 1) _56.54.pt'
    model.load_state_dict(torch.load(weight_path))

    # fbgemm 6400 all: val_loss=1.347028016738979, val_acc=0.5576819407008087
    # qnnpack 6400 all: val_loss=1.4878578570767724, val_acc=0.5015835579514825
    cal_dataloader = get_calibration_set(length=6400)

    quant_model = quantization(model, 'fx', cal_dataloader, show=True, backend='fbgemm')
    quant_model = nn.Sequential(extmel, quant_model)
    test_dataloader_list = get_valset()
    for scene_class, dataset in test_dataloader_list.items():
        val_loss, val_acc, test_device_info = validate(model, dataset,
                                                       criterion=normal_training_config['criterion'], device='cpu')
        print('{}: val_loss={}, val_acc={}'.format(scene_class, val_loss, val_acc))
