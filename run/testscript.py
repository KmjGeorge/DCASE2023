"""
随便测试点什么
"""

import torch
import torchaudio
import librosa
from dataset.datagenerator import get_tau2022
from train.normal_train import validate
from model_src.passt import passt
import torch.nn as nn
from size_cal import nessi
import matplotlib.pyplot as plt

z = torch.tensor([3, 4, 5, 3, 0, 0, 0, 0, 1])
y = torch.tensor([1, 2, 4, 4, 5, 5, 6, 4, 7])
y_ = torch.tensor([1, 6, 4, 4, 5, 5, 6, 7, 8])
print(y == y_)

device_total = {}
device_correct = {}
for device_class in set(z):
    device_total[int(device_class)] = 0
    device_correct[int(device_class)] = 0

for i in range(len(z)):
    device_total[int(z[i])] += 1

for i in range(len(z)):
    if (y == y_)[i]:
        device_correct[int(z[i])] += 1

print('total', device_total)
print('correct', device_correct)
for (device_class, correct_num), (_, total_num) in zip(device_correct.items(), device_total.items()):
    print('类别{}'.format(device_class), '准确率{}'.format(correct_num/total_num))
