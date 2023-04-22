"""
随便测试点什么
"""

import torch
from dataset.datagenerator import get_tau2022
from train.normal_train import validate
from model_src.passt import passt
import torch.nn as nn
from size_cal import nessi
import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('D:\Datasets\TAU-urban-acoustic-scenes-2022-mobile-development-reassembled\evaluation_setup\\fold1_train.csv')
df2 = pd.read_csv('D:\Datasets\TAU-urban-acoustic-scenes-2022-mobile-development-reassembled\meta.csv')
for filename in df1['filename']:
    device = df2[df2['filename'] == filename]['source_label'].values[0]
    print(filename, device)



