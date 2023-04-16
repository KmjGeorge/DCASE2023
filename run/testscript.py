'''
随便测试点什么
'''

import torch

from dataset.datagenerator import get_tau2022
from train.normal_train import validate
from model_src.passt import passt
import torch.nn as nn
from size_cal import nessi
import matplotlib.pyplot as plt
import pandas as pd


# model = passt()
# weights = torch.load('../model_weights/passt_tau2022_random_slicing_augment_mixup_epoch3.pt')
# model.load_state_dict(weights)
# nessi.get_model_size(model, 'torch', input_size=(1, 32000))
#
# train_dataloader, test_dataloader = get_tau2022()
# criterion = nn.CrossEntropyLoss()
# validate(model, test_dataloader, criterion=criterion)


def show_accloss(loss_list, acc_list, val_loss_list, val_acc_list):
    x = [i + 1 for i in range(len(loss_list))]

    plt.plot(x, loss_list, label='training loss')
    plt.plot(x, val_loss_list, label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(x, acc_list, label='training acc')
    plt.plot(x, val_acc_list, label='validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()


df = pd.read_csv('../logs/passt_tau2022_augment_mixup_mixstyle_logs.csv')
print(df.head())
