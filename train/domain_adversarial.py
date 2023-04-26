import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.datagenerator import TAU2022_DEVICES, TAU2022_DEVICES_INVERT
from optim.scheduler import GradualWarmupScheduler
from tqdm import tqdm
from dataset.augmentation import mixup, cutmix
import pandas as pd
from train.normal_train import validate
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义域对抗训练
def da_train(model, train_loader, test_loader, da_training_conf, mixup_conf,
             save=True):
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []
    test_device_info_list_a = []
    test_device_info_list_b = []
    test_device_info_list_c = []
    test_device_info_list_s1 = []
    test_device_info_list_s2 = []
    test_device_info_list_s3 = []
    test_device_info_list_s4 = []
    test_device_info_list_s5 = []
    test_device_info_list_s6 = []

    TASK_NAME = da_training_conf['task_name']
    MAX_EPOCH = da_training_conf['epoch']
    START_EPOCH = da_training_conf['start_epoch']
    OPTIM_CONF = da_training_conf['optim_config']
    SCHEDULER_COS_CONFIG = da_training_conf['scheduler_cos_config']
    SCHEDULER_WARMUP_CONFIG = da_training_conf['scheduler_warmup_config']
    LOSS_DEVICE_WEIGHT = da_training_conf['loss_device_weight']
    criterion = da_training_conf['criterion']

    # 先逐步增加至初始学习率，然后使用余弦退火
    optimizer = OPTIM_CONF['name'](model.parameters(), lr=OPTIM_CONF['lr'], weight_decay=OPTIM_CONF['weight_decay'])
    scheduler_cos = CosineAnnealingLR(optimizer,
                                      T_max=MAX_EPOCH - SCHEDULER_WARMUP_CONFIG['total_epoch'],
                                      eta_min=SCHEDULER_COS_CONFIG['eta_min'])
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=SCHEDULER_WARMUP_CONFIG['multiplier'],
                                              total_epoch=SCHEDULER_WARMUP_CONFIG['total_epoch'],
                                              after_scheduler=scheduler_cos)
    scheduler_warmup.step()  # 学习率从0开始，跳过第一轮

    for i in range(MAX_EPOCH):
        epoch_loss, epoch_acc = da_train_per_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler_warmup,
            start_epoch=START_EPOCH,
            epoch=i,
            epochs=MAX_EPOCH,
            save_name=TASK_NAME,
            mixup_conf=mixup_conf,
            loss_device_weight=LOSS_DEVICE_WEIGHT,
            save=save)
        # 每轮验证一次
        val_epoch_loss, val_epoch_acc, test_device_info = da_validate(
            model=model,
            test_loader=test_loader,
            criterion=criterion)

        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        val_loss_list.append(val_epoch_loss)
        val_acc_list.append(val_epoch_acc)
        test_device_info_list_a.append(test_device_info['a'][2])
        test_device_info_list_b.append(test_device_info['b'][2])
        test_device_info_list_c.append(test_device_info['c'][2])
        test_device_info_list_s1.append(test_device_info['s1'][2])
        test_device_info_list_s2.append(test_device_info['s2'][2])
        test_device_info_list_s3.append(test_device_info['s3'][2])
        test_device_info_list_s4.append(test_device_info['s4'][2])
        test_device_info_list_s5.append(test_device_info['s5'][2])
        test_device_info_list_s6.append(test_device_info['s6'][2])
        if save:  # 保存训练日志
            logs = pd.DataFrame({'loss': loss_list,
                                 'acc': acc_list,
                                 'val_loss': val_loss_list,
                                 'val_acc': val_acc_list,
                                 'val_A_acc': test_device_info_list_a,
                                 'val_B_acc': test_device_info_list_b,
                                 'val_C_acc': test_device_info_list_c,
                                 'val_S1_acc': test_device_info_list_s1,
                                 'val_S2_acc': test_device_info_list_s2,
                                 'val_S3_acc': test_device_info_list_s3,
                                 'val_S4_acc': test_device_info_list_s4,
                                 'val_S5_acc': test_device_info_list_s5,
                                 'val_S6_acc': test_device_info_list_s6,
                                 })
            logs.to_csv('../logs/{}_logs.csv'.format(TASK_NAME), index=True)
    print('==========Finished Training===========')
    return loss_list, acc_list, val_loss_list, val_acc_list


def da_train_per_epoch(model, train_loader, criterion, optimizer, scheduler, start_epoch, epoch, epochs, save_name,
                       mixup_conf, loss_device_weight,
                       save=True):
    correct = 0
    total = 0
    sum_loss = 0.0

    model.train()

    loop = tqdm(train_loader)
    for x, y, z in loop:
        x = x.to(device)
        y = y.long().to(device)
        z = z.long().to(device)
        z_domain = z.clone()
        for i in range(len(z)):  # 所有a设备为源域，其他为目标域
            if z[i] != 0:
                z_domain[i] = 1

        y_pred_nomix, z_pred_nomix = model(x)  # 输出为两个，一个是class标签，一个是设备标签
        if not mixup_conf['enable']:
            optimizer.zero_grad()
            y_pred = y_pred_nomix
            z_pred = z_pred_nomix
            loss_class = criterion(y_pred, y)
            loss_device = criterion(z_pred, z_domain)
            loss = loss_class + loss_device_weight * loss_device
        else:
            if np.random.rand(1) < mixup_conf['p']:
                if mixup_conf['cut']:  # 使用cutmix
                    x_mix, y_a, y_b, lamb = cutmix(x, y, mixup_conf['alpha'])
                    _, z_a, z_b, lamb = cutmix(x, z_domain, mixup_conf['alpha'])
                else:
                    x_mix, y_a, y_b, lamb = mixup(x, y, mixup_conf['alpha'])
                    _, z_a, z_b, lamb = mixup(x, z_domain, mixup_conf['alpha'])
                x_mix = x_mix.to(device)
                y_a = y_a.to(device)
                y_b = y_b.to(device)
                z_a = z_a.to(device)
                z_b = z_b.to(device)

                optimizer.zero_grad()
                y_pred, z_pred = model(x_mix)
                loss_class = lamb * criterion(y_pred, y_a) + (1 - lamb) * criterion(y_pred, y_b)
                loss_device = lamb * criterion(z_pred, z_a) + (1 - lamb) * criterion(z_pred, z_b)
                loss = loss_class + loss_device_weight * loss_device
            else:
                optimizer.zero_grad()
                y_pred = y_pred_nomix
                z_pred = z_pred_nomix
                loss_class = criterion(y_pred, y)
                loss_device = criterion(z_pred, z_domain)
                loss = loss_class + loss_device_weight * loss_device

        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']

        # 计算训练loss和acc（每批）
        with torch.no_grad():
            y_ = torch.argmax(y_pred_nomix, dim=1)
            correct += (y_ == y).sum().item()
            total += y.size(0)
            sum_loss += loss.item()
            running_loss = sum_loss / total
            running_acc = correct / total

        # 输出训练信息
        loop.set_description(f'Epoch [{start_epoch + epoch + 1}/{start_epoch + epochs}]')
        loop.set_postfix(lr=lr, loss=running_loss, acc=running_acc)
    scheduler.step()  # 更新学习率
    epoch_loss = sum_loss / total
    epoch_acc = correct / total
    if save:
        torch.save(model.state_dict(),
                   "../model_weights/{}_epoch{}.pt".format(save_name, start_epoch + epoch + 1))  # 每轮保存一次参数
    return epoch_loss, epoch_acc


def da_validate(model, test_loader, criterion):
    test_correct = 0
    test_total = 0
    test_sum_loss = 0

    test_device_info = {k: [0, 0, 0.0] for k in TAU2022_DEVICES.keys()}  # key:设备标签 v: [total_num, correct_num, acc]

    model.eval()

    with torch.no_grad():
        loop = tqdm(test_loader, desc='Validation')
        for x, y, z in loop:
            x = x.to(device)
            y = y.long().to(device)
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            y_ = torch.argmax(y_pred, dim=1)
            test_correct += (y_ == y).sum().item()
            test_total += y.size(0)
            test_sum_loss += loss.item()
            test_running_loss = test_sum_loss / test_total
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

    test_epoch_loss = test_sum_loss / test_total
    test_epoch_acc = test_correct / test_total

    model.train()

    return test_epoch_loss, test_epoch_acc, test_device_info