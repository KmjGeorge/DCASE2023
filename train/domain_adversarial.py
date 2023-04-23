import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from optim.scheduler import GradualWarmupScheduler
from tqdm import tqdm
from dataset.augmentation import mixup, cutmix
import pandas as pd
from train.normal_train import validate
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义域对抗训练
def da_train(model, train_loader, test_loader, start_epoch, normal_training_conf, mixup_conf,
          save=True):
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []

    TASK_NAME = normal_training_conf['task_name']
    MAX_EPOCH = normal_training_conf['epoch']
    OPTIM_CONF = normal_training_conf['optim_config']
    SCHEDULER_COS_CONFIG = normal_training_conf['scheduler_cos_config']
    SCHEDULER_WARMUP_CONFIG = normal_training_conf['scheduler_warmup_config']
    criterion = normal_training_conf['criterion']

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
            start_epoch=start_epoch,
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
        if save:  # 保存训练日志
            logs = pd.DataFrame({'loss': loss_list,
                                 'acc': acc_list,
                                 'val_loss': val_loss_list,
                                 'val_acc': val_acc_list,
                                 'val_A_acc': test_device_info['a'][2],
                                 'val_B_acc': test_device_info['b'][2],
                                 'val_C_acc': test_device_info['c'][2],
                                 'val_S1_acc': test_device_info['s1'][2],
                                 'val_S2_acc': test_device_info['s2'][2],
                                 'val_S3_acc': test_device_info['s3'][2],
                                 'val_S4_acc': test_device_info['s4'][2],
                                 'val_S5_acc': test_device_info['s5'][2],
                                 'val_S6_acc': test_device_info['s6'][2],
                                 })
            logs.to_csv('../logs/{}_logs.csv'.format(TASK_NAME), index=True)
    print('==========Finished Training===========')
    return loss_list, acc_list, val_loss_list, val_acc_list


def da_train_per_epoch(model, train_loader, criterion, optimizer, scheduler, start_epoch, epoch, epochs, save_name,
                    mixup_conf, gamma,
                    save=True):
    correct = 0
    total = 0
    sum_loss = 0.0

    model.train()

    loop = tqdm(train_loader)
    for x, y, z in loop:
        x = x.to(device)
        y = y.long().to(device)
        y_pred_nomix, z_pred_nomix = model(x)  # 输出为两个，一个是class标签，一个是设备标签
        if not mixup_conf['enable']:
            optimizer.zero_grad()
            y_pred = y_pred_nomix
            z_pred = z_pred_nomix
            loss_class = criterion(y_pred, y)
            loss_device = criterion(z_pred, z)
            loss = loss_class + gamma*loss_device
        else:
            if np.random.rand(1) < mixup_conf['p']:
                if mixup_conf['cut']:  # 使用cutmix
                    x_mix, y_a, y_b, lamb = cutmix(x, y, mixup_conf['alpha'])
                else:
                    x_mix, y_a, y_b, lamb = mixup(x, y, mixup_conf['alpha'])
                x_mix = x_mix.to(device)
                y_a = y_a.to(device)
                y_b = y_b.to(device)

                optimizer.zero_grad()
                y_pred, z_pred = model(x_mix)
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

