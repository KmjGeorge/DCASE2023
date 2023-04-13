'''
随便测试点什么
'''

import torch

from dataset.datagenerator import get_tau2022
from train.normal_train import validate
from model_src.passt import passt
import torch.nn as nn
from size_cal import nessi
# model = passt()
# weights = torch.load('../model_weights/passt_tau2022_random_slicing_augment_mixup_epoch3.pt')
# model.load_state_dict(weights)
# nessi.get_model_size(model, 'torch', input_size=(1, 32000))
#
# train_dataloader, test_dataloader = get_tau2022()
# criterion = nn.CrossEntropyLoss()
# validate(model, test_dataloader, criterion=criterion)



