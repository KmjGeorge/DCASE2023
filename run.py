import torch
import dataset.datagenerator
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
from size_cal import nessi
from model_src.mobilevit import mobileast_xxs, mobileast_s
from model_src.rfr_cnn import RFR_CNN
from model_src.cp_resnet import cp_resnet
from dataset.spectrum import ExtractMel
import train.normal_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_accloss(loss_list, acc_list, val_loss_list, val_acc_list, name):
    x = [i + 1 for i in range(len(loss_list))]

    plt.plot(x, loss_list, label='training loss')
    plt.plot(x, val_loss_list, label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../figure/{}_loss.jpg'.format(name, name))
    plt.show()

    plt.plot(x, acc_list, label='training acc')
    plt.plot(x, val_acc_list, label='validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig('../figure/{}_acc.jpg'.format(name, name))
    plt.show()


def save_logs(loss_list, acc_list, val_loss_list, val_acc_list):
    logs = pd.DataFrame({'loss': loss_list,
                         'acc': acc_list,
                         'val_loss': val_loss_list,
                         'val_acc': val_acc_list}
                        )
    logs.to_csv('../logs/{}_logs.csv'.format(TASK_NAME), index_label=True)
    show_accloss(loss_list, acc_list, val_loss_list, val_acc_list, name=TASK_NAME)


if __name__ == '__main__':
    '''0. 修改以下配置'''
    from dataset.dataconfig import dataset_config, spectrum_config
    from train.trainingconfig import training_config

    '''1. 读取参数'''
    MAX_EPOCH = training_config['epoch']
    TASK_NAME = training_config['task_name']
    MIXUP_ALPHA = training_config['mixup_alpha']
    CHOOSE_MODEL = training_config['model']
    DATASET_NAME = dataset_config['name']

    '''2. 获取模型'''
    # 频谱特征提取和增强在dataset.spectrum.ExtractMel，作为网络的一个输入层使用
    if CHOOSE_MODEL == 'cp_resnet':
        model = nn.Sequential(ExtractMel(**spectrum_config), cp_resnet()).to(device)
    elif CHOOSE_MODEL == 'mobileast_s':
        model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_s()).to(device)
    elif CHOOSE_MODEL == 'mobileast_xxs':
        model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_xxs()).to(device)
    elif CHOOSE_MODEL == 'RFR-CNN':
        model = nn.Sequential(ExtractMel(**spectrum_config), RFR_CNN().to(device))
    else:
        raise '未定义的模型！'

    '''3. 计算模型大小，需指定输入形状 (batch, sr*time) '''
    nessi.get_model_size(model, 'torch', input_size=(1, spectrum_config['sr'] * 4))

    '''4. 获取数据集'''
    # 注意如果使用tau2022_random_slicing，请先在dataset.datagenerator中生成该数据集
    if DATASET_NAME == 'TAU2022_RANDOM_SLICING' or DATASET_NAME == 'tau2022_random_slicing':
        train_dataloader, test_dataloader = dataset.datagenerator.get_tau2022_random_slicing()
    elif DATASET_NAME == 'TAU2022' or DATASET_NAME == 'tau2022':
        train_dataloader, test_dataloader = dataset.datagenerator.get_tau2022()
    elif DATASET_NAME == 'urbansound8k' or DATASET_NAME == 'URBANSOUND8K':
        train_dataloader, test_dataloader = dataset.datagenerator.get_urbansound8k(fold_shuffle=True)
    else:
        raise '未定义的数据集！'

    '''5. 训练并显示曲线 训练参数存放于model_weights 文件名为task_name名
          曲线存放于figure文件夹 文件名为task_name名'''
    loss_list, acc_list, val_loss_list, val_acc_list = train.normal_train.train(model, train_dataloader,
                                                                                test_dataloader,
                                                                                epochs=MAX_EPOCH, save_name=TASK_NAME,
                                                                                mixup_alpha=MIXUP_ALPHA,
                                                                                save=True)
    '''6. 保存日志 存放于logs文件夹 文件名为task_name名'''
    save_logs(loss_list, acc_list, val_loss_list, val_acc_list)
