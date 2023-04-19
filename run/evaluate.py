import torch
import dataset.datagenerator
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import random
import numpy as np
from size_cal import nessi
from model_src.mobilevit import mobileast_xxs, mobileast_s, mobileast_light
from model_src.rfr_cnn import RFR_CNN
from model_src.cp_resnet import cp_resnet
from model_src.passt import passt
from model_src.acdnet import GetACDNetModel
from dataset.spectrum import ExtractMel
import train.normal_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    from configs.dataconfig import dataset_config, spectrum_config
    from configs.trainingconfig import normal_training_config
    from configs.mixup import mixup_config
    from configs.mixstyle import mixstyle_config
    # 固定种子
    setup_seed(200)

    from dataset.datagenerator import TAU2022
    from torch.utils.data import DataLoader
    from train.normal_train import validate
    import os
    test_h5 = os.path.join(dataset_config['h5path'], 'tau2022_test.h5')
    test_dataloader = DataLoader(TAU2022(test_h5), batch_size=dataset_config['batch_size'], shuffle=False)
    model = passt(mixstyle_conf=mixstyle_config, pretrained=False, n_classes=10).to(device)
    model.load_state_dict(torch.load('../model_weights/passt_tau2022_random_slicing_augment_fpatchout=6_mixup(alpha=0.3)_mixstyle(alpha=0.3,p=0.6)_valacc=59.87.pt'))
    val_loss, val_acc = validate(model, test_dataloader, criterion=normal_training_config['criterion'])


