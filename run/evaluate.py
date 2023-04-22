import torch
import dataset.datagenerator
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import random
import numpy as np
from model_src.passt import passt
from model_src.mobilevit import mobileast_light
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
    import soundfile
    from dataset.spectrum import plot_waveform

    test_h5 = os.path.join(dataset_config['h5path'], 'tau2022_test.h5')
    test_dataloader = DataLoader(TAU2022(test_h5), batch_size=dataset_config['batch_size'], shuffle=False)
    # i = 0
    # for x, y in test_dataloader:
    #     if i == 1:
    #         break
    #     for j in range(64):
    #         plot_waveform(torch.unsqueeze(x[j], dim=0), 32000)
    #         soundfile.write('../raw/audio{}_{}.wav'.format(i, j), x[j], 32000)
    #     i += 1
    from dataset.spectrum import ExtractMel
    #model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_light(mixstyle_conf=mixstyle_config)).to(device)
    model = passt(mixstyle_config, pretrained_local=False, n_classes=10).to(device)
    weights = torch.load(
        '../model_weights/passt_tau2022_random_slicing_augment_fpatchout=6_mixup(alpha=0.3)_mixstyle(alpha=0.3,p=0.6)_valacc=59.87.pt')
    model.load_state_dict(weights)
    val_loss, val_acc = validate(model, test_dataloader, criterion=normal_training_config['criterion'])
