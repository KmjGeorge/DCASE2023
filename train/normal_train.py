import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from optim.scheduler import GradualWarmupScheduler
from optim.scheduler import ExpWarmupLinearDownScheduler
from tqdm import tqdm
from dataset.augmentation import mixup, cutmix
import pandas as pd
from dataset.datagenerator import TAU2022_DEVICES, TAU2022_DEVICES_INVERT, TAU2022_CLASSES
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义普通的训练过程
def train(model, train_loader, test_loader, normal_training_conf, mixup_conf,
          save=True):
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []
    val_acc_a_list = []
    val_acc_b_list = []
    val_acc_c_list = []
    val_acc_s1_list = []
    val_acc_s2_list = []
    val_acc_s3_list = []
    val_acc_s4_list = []
    val_acc_s5_list = []
    val_acc_s6_list = []

    TASK_NAME = normal_training_conf['task_name']
    MAX_EPOCH = normal_training_conf['epoch']
    START_EPOCH = normal_training_conf['start_epoch']
    OPTIM_CONF = normal_training_conf['optim_config']
    SCHEDULER_WARMUP_CONFIG = normal_training_conf['scheduler_warmup_config']
    SCHEDULER_DOWN_CONFIG = normal_training_conf['scheduler_down_config']
    criterion = normal_training_conf['criterion']

    optimizer = OPTIM_CONF['name'](model.parameters(), lr=OPTIM_CONF['lr'], weight_decay=OPTIM_CONF['weight_decay'])
    # 先指数增长至初始学习率，然后使用线性衰减
    scheduler_exp_warmup_linear_down = ExpWarmupLinearDownScheduler(optimizer,
                                                                    warm_up_len=SCHEDULER_WARMUP_CONFIG['total_epoch'],
                                                                    ramp_down_start=SCHEDULER_WARMUP_CONFIG[
                                                                        'total_epoch'],
                                                                    ramp_down_len=SCHEDULER_DOWN_CONFIG['total_epoch'],
                                                                    last_lr_value=SCHEDULER_DOWN_CONFIG['eta_min'])

    '''
    # 先逐步增加至初始学习率，然后使用余弦退火
    scheduler_cos = CosineAnnealingLR(optimizer,
                                      T_max=MAX_EPOCH - SCHEDULER_WARMUP_CONFIG['total_epoch'],
                                      eta_min=SCHEDULER_COS_CONFIG['eta_min'])
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=SCHEDULER_WARMUP_CONFIG['multiplier'],
                                              total_epoch=SCHEDULER_WARMUP_CONFIG['total_epoch'],
                                              after_scheduler=scheduler_cos)
    scheduler_warmup.step()  # 学习率从0开始，跳过第一轮
    '''

    for i in range(MAX_EPOCH):
        epoch_loss, epoch_acc = train_per_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler_exp_warmup_linear_down,
            start_epoch=START_EPOCH,
            epoch=i,
            epochs=MAX_EPOCH,
            save_name=TASK_NAME,
            mixup_conf=mixup_conf,
            save=save)
        # 每轮验证一次
        val_epoch_loss, val_epoch_acc, test_device_info = validate(
            model=model,
            test_loader=test_loader,
            criterion=criterion)

        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        val_loss_list.append(val_epoch_loss)
        val_acc_list.append(val_epoch_acc)
        val_acc_a_list.append(test_device_info['a'][2])
        val_acc_b_list.append(test_device_info['b'][2])
        val_acc_c_list.append(test_device_info['c'][2])
        val_acc_s1_list.append(test_device_info['s1'][2])
        val_acc_s2_list.append(test_device_info['s2'][2])
        val_acc_s3_list.append(test_device_info['s3'][2])
        val_acc_s4_list.append(test_device_info['s4'][2])
        val_acc_s5_list.append(test_device_info['s5'][2])
        val_acc_s6_list.append(test_device_info['s6'][2])
        if save:  # 保存训练日志
            logs = pd.DataFrame({'loss': loss_list,
                                 'acc': acc_list,
                                 'val_loss': val_loss_list,
                                 'val_acc': val_acc_list,
                                 'val_A_acc': val_acc_a_list,
                                 'val_B_acc': val_acc_b_list,
                                 'val_C_acc': val_acc_c_list,
                                 'val_S1_acc': val_acc_s1_list,
                                 'val_S2_acc': val_acc_s2_list,
                                 'val_S3_acc': val_acc_s3_list,
                                 'val_S4_acc': val_acc_s4_list,
                                 'val_S5_acc': val_acc_s5_list,
                                 'val_S6_acc': val_acc_s6_list,
                                 })
            logs.to_csv('../logs/{}_logs.csv'.format(TASK_NAME), index=True)
    print('==========Finished Training===========')
    return loss_list, acc_list, val_loss_list, val_acc_list


def train_per_epoch(model, train_loader, criterion, optimizer, scheduler, start_epoch, epoch, epochs, save_name,
                    mixup_conf,
                    save=True):
    correct = 0
    total = 0
    total_batch = 0
    sum_loss = 0.0

    model.train()

    loop = tqdm(train_loader)
    for x, y, z in loop:
        x = x.to(device)
        y = y.long().to(device)
        y_pred_nomix = model(x)
        if not mixup_conf['enable']:  # 不使用mixup
            optimizer.zero_grad()
            y_pred = y_pred_nomix
            loss = criterion(y_pred, y)
        else:
            if np.random.rand(1) < mixup_conf['p']:
                if mixup_conf['cut']:  # 使用cutmix
                    mel = model[0](x)
                    mel_mix, y_a, y_b, lamb = cutmix(mel, y, mixup_conf['alpha'])
                    mel_mix = mel_mix.to(device)
                    y_a = y_a.to(device)
                    y_b = y_b.to(device)
                    y_pred = model[1](mel_mix)
                else:
                    x_mix, y_a, y_b, lamb = mixup(x, y, mixup_conf['alpha'])
                    x_mix = x_mix.to(device)
                    y_a = y_a.to(device)
                    y_b = y_b.to(device)
                    y_pred = model(x_mix)

                optimizer.zero_grad()
                loss = lamb * criterion(y_pred, y_a) + (1 - lamb) * criterion(y_pred, y_b)
            else:
                optimizer.zero_grad()
                y_pred = y_pred_nomix
                loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']

        # 计算训练loss和acc（每批）
        with torch.no_grad():
            y_ = torch.argmax(y_pred_nomix, dim=1)
            correct += (y_ == y).sum().item()
            total += y.size(0)
            total_batch += 1
            sum_loss += loss.item()
            running_loss = sum_loss / total_batch  # loss已经是按batch计算的，均值分母为batch数
            running_acc = correct / total

        # 输出训练信息
        loop.set_description(f'Epoch [{start_epoch + epoch + 1}/{start_epoch + epochs}]')
        loop.set_postfix(lr=lr, loss=running_loss, acc=running_acc)
    scheduler.step()  # 更新学习率
    epoch_loss = sum_loss / total_batch
    epoch_acc = correct / total
    if save:
        torch.save(model.state_dict(),
                   "../model_weights/{}_epoch{}.pt".format(save_name, start_epoch + epoch + 1))  # 每轮保存一次参数
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device='cuda'):
    test_correct = 0
    test_total = 0
    test_total_batch = 0
    test_sum_loss = 0

    test_device_info = {k: [0, 0, 0.0] for k in TAU2022_DEVICES.keys()}  # key:设备标签 v: [total_num, correct_num, acc]

    model.eval()

    with torch.no_grad():
        loop = tqdm(test_loader, desc='Validation')
        for x, y, z in loop:
            x = x.to(device)
            y = y.long().to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            y_ = torch.argmax(y_pred, dim=1)
            test_correct += (y_ == y).sum().item()
            test_total += y.size(0)
            test_total_batch += 1
            test_sum_loss += loss.item()
            test_running_loss = test_sum_loss / test_total_batch
            test_running_acc = test_correct / test_total

            # 计算每个设备的准确率
            for i, source_label in enumerate(z):
                key = TAU2022_DEVICES_INVERT[int(source_label)]
                test_device_info[key][0] += 1
                if (y == y_)[i]:
                    test_device_info[key][1] += 1
                test_device_info[key][2] = test_device_info[key][1] / test_device_info[key][0]

            # 输出验证信息
            loop.set_postfix(val_loss=test_running_loss, val_acc=test_running_acc,
                             A=test_device_info['a'][2],
                             B=test_device_info['b'][2],
                             C=test_device_info['c'][2],
                             S1=test_device_info['s1'][2],
                             S2=test_device_info['s2'][2],
                             S3=test_device_info['s3'][2],
                             S4=test_device_info['s4'][2],
                             S5=test_device_info['s5'][2],
                             S6=test_device_info['s6'][2],
                             )

    test_epoch_loss = test_sum_loss / test_total_batch
    test_epoch_acc = test_correct / test_total

    model.train()

    return test_epoch_loss, test_epoch_acc, test_device_info


# 带TTA的验证 aug_func_list为数据增强函数列表
def validate_with_TTA(model, test_loader, criterion, augment_func_list, device='cuda'):
    test_correct = 0
    test_total = 0
    test_total_batch = 0
    test_sum_loss = 0

    test_device_info = {k: [0, 0, 0.0] for k in TAU2022_DEVICES.keys()}  # key:设备标签 v: [total_num, correct_num, acc]

    model.eval()

    with torch.no_grad():
        loop = tqdm(test_loader, desc='Validation')
        for x, y, z in loop:
            x = x.to(device)
            y = y.long().to(device)
            y_pred = torch.zeros(size=(64, 10))
            for augment in augment_func_list:
                x_aug = augment(x)
                y_pred += model(x_aug)
            y_pred = y_pred / len(augment_func_list)
            loss = criterion(y_pred, y)
            y_ = torch.argmax(y_pred, dim=1)
            test_correct += (y_ == y).sum().item()
            test_total += y.size(0)
            test_total_batch += 1
            test_sum_loss += loss.item()
            test_running_loss = test_sum_loss / test_total_batch
            test_running_acc = test_correct / test_total

            # 计算每个设备的准确率
            for i, source_label in enumerate(z):
                key = TAU2022_DEVICES_INVERT[int(source_label)]
                test_device_info[key][0] += 1
                if (y == y_)[i]:
                    test_device_info[key][1] += 1
                test_device_info[key][2] = test_device_info[key][1] / test_device_info[key][0]

            # 输出验证信息
            loop.set_postfix(val_loss=test_running_loss, val_acc=test_running_acc,
                             A=test_device_info['a'][2],
                             B=test_device_info['b'][2],
                             C=test_device_info['c'][2],
                             S1=test_device_info['s1'][2],
                             S2=test_device_info['s2'][2],
                             S3=test_device_info['s3'][2],
                             S4=test_device_info['s4'][2],
                             S5=test_device_info['s5'][2],
                             S6=test_device_info['s6'][2],
                             )

    test_epoch_loss = test_sum_loss / test_total_batch
    test_epoch_acc = test_correct / test_total

    model.train()

    return test_epoch_loss, test_epoch_acc, test_device_info
