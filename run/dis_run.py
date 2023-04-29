import torch
import dataset.datagenerator
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import numpy as np

from model_src.cp_resnet import cp_resnet
from model_src.mobilevit import mobileast_light2, mobileast_light, mobileast_cpresnet2
from model_src.passt import passt
from size_cal import nessi
from dataset.spectrum import ExtractMel
import train.distillation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(name, wave=True):
    # wave=True 添加一个频谱特征提取层，使得网络输入可以为波形
    if name == 'cp_resnet':
        if wave:
            model = nn.Sequential(ExtractMel(**spectrum_config),
                                  cp_resnet(mixstyle_config, rho=8, s2_group=2, cut_channels_s3=36,
                                            n_blocks=(2, 1, 1))).to(device)
        else:
            model = cp_resnet(mixstyle_config, rho=8, s2_group=2, cut_channels_s3=36, n_blocks=(2, 1, 1)).to(device)
    elif name == 'mobileast_light':
        if wave:
            model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_light(mixstyle_conf=mixstyle_config)).to(
                device)
        else:
            model = mobileast_light(mixstyle_conf=mixstyle_config).to(device)
    elif name == 'mobileast_light2':
        if wave:
            model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_light2(mixstyle_conf=mixstyle_config)).to(
                device)
        else:
            model = mobileast_light2(mixstyle_conf=mixstyle_config).to(device)
    elif name == 'mobileast_cpresnet':
        if wave:
            model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_cpresnet2(mixstyle_conf=mixstyle_config)).to(
                device)
        else:
            model = mobileast_cpresnet2(mixstyle_conf=mixstyle_config).to(device)
    elif name == 'passt':
        model = passt(mixstyle_conf=mixstyle_config, pretrained_local=False, n_classes=10).to(device)
    else:
        raise '未定义的模型！'

    return model


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
    from configs.trainingconfig import distillation_config
    from configs.mixup import mixup_config
    from configs.mixstyle import mixstyle_config

    # 固定种子
    setup_seed(200)

    TEACHER_WEIGHT_PATH = distillation_config['teacher_weight_path']
    TEACHER_MODEL = distillation_config['teacher_model']
    STUDENT_MODEL = distillation_config['student_model']
    DATASET_NAME = dataset_config['name']
    TASK_NAME = distillation_config['task_name']

    '''2. 获取模型'''
    teacher = get_model(TEACHER_MODEL)
    student = get_model(STUDENT_MODEL)
    weights = torch.load(TEACHER_WEIGHT_PATH)
    # weights_new = {}
    # for k, v in weights.items():
    #     weights_new[k.replace('net.1.', 'net.')] = v
    teacher.load_state_dict(weights)

    '''3. 计算模型大小，需指定输入形状 (batch, sr*time) '''
    nessi.get_model_size(teacher, 'torch', input_size=(1, spectrum_config['sr'] * 1))
    nessi.get_model_size(student, 'torch', input_size=(1, spectrum_config['sr'] * 1))

    '''4. 获取数据集'''
    # 请先在dataset.datagenerator中生成数据集(h5形式)
    if DATASET_NAME == 'TAU2022_RANDOM_SLICING' or DATASET_NAME == 'tau2022_random_slicing':
        train_dataloader, test_dataloader = dataset.datagenerator.get_tau2022_reassembled_random_slicing()
    elif DATASET_NAME == 'TAU2022' or DATASET_NAME == 'tau2022':
        train_dataloader, test_dataloader = dataset.datagenerator.get_tau2022()
    elif DATASET_NAME == 'TAU2022_REASSEMBLED' or DATASET_NAME == 'tau2022_reassembled':
        train_dataloader, test_dataloader = dataset.datagenerator.get_tau2022_reassembled()
    else:
        raise '未定义的数据集！'

    '''5. 训练 参数存放于model_weights 文件名为training_config['task_name']名'''
    loss_list, acc_list, val_loss_list, val_acc_list = train.distillation.distillation(student=student, teacher=teacher,
                                                                                       train_loader=train_dataloader,
                                                                                       test_loader=test_dataloader,
                                                                                       distillation_conf=distillation_config,
                                                                                       mixup_conf=mixup_config,
                                                                                       save=True)
    '''6. 绘制曲线
    曲线存放于figure文件夹 文件名为training_config['task_name']名
    日志存放于logs文件夹 文件名为training_config['task_name']名'''
    show_accloss(loss_list, acc_list, val_loss_list, val_acc_list, save_name=TASK_NAME)
