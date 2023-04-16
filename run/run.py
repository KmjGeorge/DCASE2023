import torch
import dataset.datagenerator
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import random
import numpy as np
from size_cal import nessi
from model_src.mobilevit import mobileast_xxs, mobileast_s
from model_src.rfr_cnn import RFR_CNN
from model_src.cp_resnet import cp_resnet
from model_src.passt import passt
from model_src.acdnet import GetACDNetModel
from dataset.spectrum import ExtractMel
import train.normal_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_accloss(loss_list, acc_list, val_loss_list, val_acc_list, save_name):
    x = [i + 1 for i in range(len(loss_list))]

    plt.plot(x, loss_list, label='training loss')
    plt.plot(x, val_loss_list, label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../figure/{}_loss.jpg'.format(save_name))
    plt.show()

    plt.plot(x, acc_list, label='training acc')
    plt.plot(x, val_acc_list, label='validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig('../figure/{}_acc.jpg'.format(save_name))
    plt.show()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    '''1. 读取配置'''
    from configs.dataconfig import dataset_config, spectrum_config
    from configs.trainingconfig import normal_training_config
    from configs.mixup import mixup_config
    from configs.mixstyle import mixstyle_config

    # 固定种子
    setup_seed(200)

    CHOOSE_MODEL = normal_training_config['model']
    DATASET_NAME = dataset_config['name']
    TASK_NAME = normal_training_config['task_name']

    '''2. 获取模型'''
    # 频谱特征提取和增强在dataset.spectrum.ExtractMel，作为网络的一个输入层使用
    if CHOOSE_MODEL == 'cp_resnet':
        model = nn.Sequential(ExtractMel(**spectrum_config), cp_resnet(mixstyle_conf=mixstyle_config)).to(device)
    elif CHOOSE_MODEL == 'mobileast_s':
        model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_s(mixstyle_conf=mixstyle_config)).to(device)
    elif CHOOSE_MODEL == 'mobileast_xxs':
        model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_xxs(mixstyle_conf=mixstyle_config)).to(device)
    elif CHOOSE_MODEL == 'rfr-cnn':
        model = nn.Sequential(ExtractMel(**spectrum_config), RFR_CNN().to(device))
    elif CHOOSE_MODEL == 'passt':
        model = passt(mixstyle_conf=mixstyle_config, n_classes=10).to(device)
    elif CHOOSE_MODEL == 'acdnet':
        model = GetACDNetModel(input_len=32000, nclass=10, sr=spectrum_config['sr'])
    else:
        raise '未定义的模型！'

    '''3. 计算模型大小，需指定输入形状 (batch, sr*time) '''
    nessi.get_model_size(model, 'torch', input_size=(1, spectrum_config['sr'] * 1))

    '''4. 获取数据集'''
    # 请先在dataset.datagenerator中生成数据集(h5形式)
    if DATASET_NAME == 'TAU2022_RANDOM_SLICING' or DATASET_NAME == 'tau2022_random_slicing':
        train_dataloader, test_dataloader = dataset.datagenerator.get_tau2022_reassembled_random_slicing()
    elif DATASET_NAME == 'TAU2022' or DATASET_NAME == 'tau2022':
        train_dataloader, test_dataloader = dataset.datagenerator.get_tau2022()
    elif DATASET_NAME == 'urbansound8k' or DATASET_NAME == 'URBANSOUND8K':
        train_dataloader, test_dataloader = dataset.datagenerator.get_urbansound8k(fold_shuffle=True)
    elif DATASET_NAME == 'TAU2022_REASSEMBLED' or DATASET_NAME == 'tau2022_reassembled':
        train_dataloader, test_dataloader = dataset.datagenerator.get_tau2022_reassembled()
    else:
        raise '未定义的数据集！'

    '''5. 训练 参数存放于model_weights 文件名为training_config['task_name']名'''
    # 如果是断点续训，载入参数
    # weights = torch.load('../model_weights/passt_tau2022_random_slicing_augment_mixup_epoch3.pt')
    # model.load_state_dict(weights)
    loss_list, acc_list, val_loss_list, val_acc_list = train.normal_train.train(model, train_dataloader,
                                                                                test_dataloader,
                                                                                start_epoch=0,
                                                                                normal_training_conf=normal_training_config,
                                                                                mixup_conf=mixup_config,
                                                                                save=True)
    '''6. 绘制曲线
    曲线存放于figure文件夹 文件名为training_config['task_name']名
    日志存放于logs文件夹 文件名为training_config['task_name']名'''
    show_accloss(loss_list, acc_list, val_loss_list, val_acc_list, save_name=TASK_NAME)
