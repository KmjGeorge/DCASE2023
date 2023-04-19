import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from optim.scheduler import GradualWarmupScheduler
from tqdm import tqdm
from dataset.augmentation import mixup
from train.normal_train import validate
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def distillation(student, teacher, train_loader, test_loader, start_epoch, distillation_conf, mixup_conf,
                 save=True):
    TASK_NAME = distillation_conf['task_name']
    MAX_EPOCH = distillation_conf['epoch']
    OPTIM_CONF = distillation_conf['optim_config']
    SCHEDULER_COS_CONFIG = distillation_conf['scheduler_cos_config']
    SCHEDULER_WARMUP_CONFIG = distillation_conf['scheduler_warmup_config']
    hard_criterion = distillation_conf['hard_criterion']
    soft_criterion = distillation_conf['soft_criterion']
    T = distillation_conf['T']
    alpha = distillation_conf['alpha']

    student.train()

    optimizer = OPTIM_CONF['name'](student.parameters(), lr=OPTIM_CONF['lr'], weight_decay=OPTIM_CONF['weight_decay'])
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=SCHEDULER_COS_CONFIG['eta_min'])
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=SCHEDULER_WARMUP_CONFIG['multiplier'],
                                              total_epoch=SCHEDULER_WARMUP_CONFIG['total_epoch'],
                                              after_scheduler=scheduler_cos)
    scheduler_warmup.step()  # 学习率从0开始，跳过第一轮

    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []

    for i in range(MAX_EPOCH):
        epoch_loss, epoch_acc = distillation_per_epoch(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            hard_criterion=hard_criterion,
            soft_criterion=soft_criterion,
            optimizer=optimizer,
            scheduler=scheduler_warmup,
            T=T,
            alpha=alpha,
            start_epoch=start_epoch,
            epoch=i,
            epochs=MAX_EPOCH,
            save_name=TASK_NAME,
            mixup_conf=mixup_conf,
            save=save)
        # 每轮验证一次
        val_epoch_loss, val_epoch_acc = validate(
            model=student,
            test_loader=test_loader,
            criterion=hard_criterion)

        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        val_loss_list.append(val_epoch_loss)
        val_acc_list.append(val_epoch_acc)
        if save:  # 保存训练日志
            logs = pd.DataFrame({'loss': loss_list,
                                 'acc': acc_list,
                                 'val_loss': val_loss_list,
                                 'val_acc': val_acc_list}
                                )
            logs.to_csv('../logs/{}_logs.csv'.format(TASK_NAME), index=True)
    print('==========Finished Training===========')

    return loss_list, acc_list, val_loss_list, val_acc_list


def distillation_per_epoch(student, teacher, train_loader, hard_criterion, soft_criterion, optimizer, scheduler, T,
                           alpha,
                           start_epoch, epoch, epochs,
                           save_name,
                           mixup_conf,
                           save=True):
    correct = 0
    total = 0
    sum_loss = 0.0
    student.train()

    loop = tqdm(train_loader)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        # 计算hard_loss
        y_pred_nomix = student(x)
        if not mixup_conf['enable']:  # 不使用mixup
            optimizer.zero_grad()
            hard_loss = hard_criterion(y_pred_nomix, y)
        else:
            x_mix, y_a, y_b, lamb = mixup(x, y, mixup_conf['alpha'])
            x_mix = x_mix.to(device)
            y_a = y_a.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            y_pred_mix = student(x_mix)
            hard_loss = lamb * hard_criterion(y_pred_mix, y_a) + (1 - lamb) * hard_criterion(y_pred_mix, y_b)

        # 计算soft_loss
        with torch.no_grad():
            teacher_pred = teacher(x)
        y_pred_log_softmax = F.log_softmax(y_pred_nomix, dim=1)  # 学生不使用温度(参考arXiv:2203.06760v1)
        teacher_pred_softmax = F.softmax(teacher_pred / T, dim=1)
        soft_loss = soft_criterion(y_pred_log_softmax,
                                   teacher_pred_softmax)  # 注意torch的KL散度中，模型预测结果取log_softmax，而教师预测结果取softmax(如果log_target=False)
        
        # 总loss
        loss = alpha * hard_loss + (1 - alpha) * soft_loss
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']

        # 计算训练loss和acc（每批）
        with torch.no_grad():
            if not mixup_conf['enable']:
                y_ = torch.argmax(y_pred_nomix, dim=1)
            else:
                y_ = torch.argmax(y_pred_nomix, dim=1)  # 使用非mixup的数据计算acc
            correct += (y_ == y).sum().item()
            total += y.size(0)
            sum_loss += loss.item()
            running_loss = sum_loss / total
            running_acc = correct / total

        # 输出训练信息
        loop.set_description(f'Epoch [{start_epoch + epoch + 1}/{epochs}]')
        loop.set_postfix(lr=lr, loss=running_loss, acc=running_acc)
    scheduler.step()  # 更新学习率
    epoch_loss = sum_loss / total
    epoch_acc = correct / total

    if save:
        torch.save(student.state_dict(),
                   "../model_distillation_weights/{}_epoch{}.pt".format(save_name, start_epoch + epoch + 1))  # 每轮保存一次参数

    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion):
    test_correct = 0
    test_total = 0
    test_sum_loss = 0
    model.eval()

    with torch.no_grad():
        loop = tqdm(test_loader, desc='Validation')
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            y_ = torch.argmax(y_pred, dim=1)
            test_correct += (y_ == y).sum().item()
            test_total += y.size(0)
            test_sum_loss += loss.item()
            test_running_loss = test_sum_loss / test_total
            test_running_acc = test_correct / test_total

            # 输出验证信息
            loop.set_postfix(val_loss=test_running_loss, val_acc=test_running_acc)

    test_epoch_loss = test_sum_loss / test_total
    test_epoch_acc = test_correct / test_total
    model.train()

    return test_epoch_loss, test_epoch_acc
