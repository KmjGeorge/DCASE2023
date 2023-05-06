import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from optim.scheduler import GradualWarmupScheduler, ExpWarmupLinearDownScheduler
from tqdm import tqdm
from dataset.augmentation import mixup, cutmix
import numpy as np
from train.normal_train import validate
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def distillation(student, teacher, train_loader, test_loader, distillation_conf, mixup_conf, mode='normal',
                 save=True):
    TASK_NAME = distillation_conf['task_name']
    MAX_EPOCH = distillation_conf['epoch']
    START_EPOCH = distillation_conf['start_epoch']
    OPTIM_CONF = distillation_conf['optim_config']
    SCHEDULER_WARMUP_CONFIG = distillation_conf['scheduler_warmup_config']
    SCHEDULER_DOWN_CONFIG = distillation_conf['scheduler_down_config']
    hard_criterion = distillation_conf['hard_criterion']
    soft_criterion = distillation_conf['soft_criterion']
    T = distillation_conf['T']
    alpha = distillation_conf['alpha']
    beta = distillation_conf['beta']

    student.train()

    optimizer = OPTIM_CONF['name'](student.parameters(), lr=OPTIM_CONF['lr'], weight_decay=OPTIM_CONF['weight_decay'])

    # scheduler_cos = CosineAnnealingLR(optimizer,
    #                                   T_max=MAX_EPOCH - SCHEDULER_WARMUP_CONFIG['total_epoch'],
    #                                   eta_min=SCHEDULER_COS_CONFIG['eta_min'])
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=SCHEDULER_WARMUP_CONFIG['multiplier'],
    #                                           total_epoch=SCHEDULER_WARMUP_CONFIG['total_epoch'],
    #                                           after_scheduler=scheduler_cos)
    # scheduler_warmup.step()  # 学习率从0开始，跳过第一轮
    scheduler_exp_warmup_linear_down = ExpWarmupLinearDownScheduler(optimizer,
                                                                    warm_up_len=SCHEDULER_WARMUP_CONFIG['total_epoch'],
                                                                    ramp_down_start=SCHEDULER_WARMUP_CONFIG[
                                                                        'total_epoch'],
                                                                    ramp_down_len=SCHEDULER_DOWN_CONFIG['total_epoch'],
                                                                    last_lr_value=SCHEDULER_DOWN_CONFIG['eta_min'])

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

    for i in range(MAX_EPOCH):
        if mode == 'dkd':
            epoch_loss, epoch_acc = dkd_per_epoch(
                student=student,
                teacher=teacher,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler_exp_warmup_linear_down,
                T=T,
                alpha=alpha,
                beta=beta,
                start_epoch=START_EPOCH,
                epoch=i,
                epochs=MAX_EPOCH,
                save_name=TASK_NAME,
                mixup_conf=mixup_conf,
                save=save
            )
        elif mode == 'normal':
            epoch_loss, epoch_acc = distillation_per_epoch(
                student=student,
                teacher=teacher,
                train_loader=train_loader,
                hard_criterion=hard_criterion,
                soft_criterion=soft_criterion,
                optimizer=optimizer,
                scheduler=scheduler_exp_warmup_linear_down,
                T=T,
                alpha=alpha,
                start_epoch=START_EPOCH,
                epoch=i,
                epochs=MAX_EPOCH,
                save_name=TASK_NAME,
                mixup_conf=mixup_conf,
                save=save)
        else:
            raise '未定义的蒸馏!'
        # 每轮验证一次
        val_epoch_loss, val_epoch_acc, test_device_info = validate(
            model=student,
            test_loader=test_loader,
            criterion=hard_criterion)

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


def distillation_per_epoch(student, teacher, train_loader, hard_criterion, soft_criterion, optimizer, scheduler, T,
                           alpha,
                           start_epoch, epoch, epochs,
                           save_name,
                           mixup_conf,
                           save=True):
    correct = 0
    total = 0
    total_batch = 0
    sum_loss = 0.0
    student.train()

    loop = tqdm(train_loader)
    for x, y, _ in loop:
        x = x.to(device)
        y = y.long().to(device)
        optimizer.zero_grad()
        # 计算hard_loss
        y_pred_nomix = student(x)
        if not mixup_conf['enable']:  # 不使用mixup
            hard_loss = hard_criterion(y_pred_nomix, y)
        else:
            if np.random.rand(1) < mixup_conf['p']:
                if mixup_conf['cut']:
                    mel = student[0](x)   # cutmix是对特征图的处理，先提取mel谱
                    mel_mix, y_a, y_b, lamb = cutmix(mel, y, mixup_conf['alpha'])
                    mel_mix = mel_mix.to(device)
                    y_a = y_a.to(device)
                    y_b = y_b.to(device)
                    y_pred_mix = student[1](mel_mix)
                else:
                    x_mix, y_a, y_b, lamb = mixup(x, y, mixup_conf['alpha'])
                    x_mix = x_mix.to(device)
                    y_a = y_a.to(device)
                    y_b = y_b.to(device)
                    y_pred_mix = student(x_mix)
                hard_loss = lamb * hard_criterion(y_pred_mix, y_a) + (1 - lamb) * hard_criterion(y_pred_mix, y_b)
            else:
                hard_loss = hard_criterion(y_pred_nomix, y)

        # 计算soft_loss
        with torch.no_grad():
            teacher_pred = teacher(x)
        y_pred_log_softmax = F.log_softmax(y_pred_nomix / T, dim=1)
        teacher_pred_softmax = F.softmax(teacher_pred / T, dim=1)
        soft_loss = soft_criterion(y_pred_log_softmax,
                                   teacher_pred_softmax)  # 注意torch的KL散度中，模型预测结果取log_softmax，而教师预测结果取softmax(如果log_target=False)

        # 总loss
        loss = alpha * soft_loss + hard_loss
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']

        # 计算训练loss和acc（每批）
        with torch.no_grad():
            y_ = torch.argmax(y_pred_nomix, dim=1)  # 使用非mixup的数据计算acc
            correct += (y_ == y).sum().item()
            total += y.size(0)
            total_epoch += 1
            sum_loss += loss.item()
            running_loss = sum_loss / total_batch
            running_acc = correct / total

        # 输出训练信息
        loop.set_description(f'Epoch [{start_epoch + epoch + 1}/{epochs + start_epoch}]')
        loop.set_postfix(lr=lr, loss=running_loss, acc=running_acc)
    scheduler.step()  # 更新学习率
    epoch_loss = sum_loss / total_batch
    epoch_acc = correct / total

    if save:
        torch.save(student.state_dict(),
                   "../model_weights/{}_epoch{}.pt".format(save_name, start_epoch + epoch + 1))  # 每轮保存一次参数

    return epoch_loss, epoch_acc


def dkd_per_epoch(student, teacher, train_loader, T, alpha, beta, start_epoch, optimizer, scheduler, epoch, epochs,
                  save_name, mixup_conf, save):
    """
    T: 蒸馏温度
    alpha: TCKD损失系数
    beta: NCKD损失系数
    """
    correct = 0
    total = 0
    total_batch = 0
    sum_loss = 0.0
    student.train()
    loop = tqdm(train_loader)
    for x, y, _ in loop:
        x = x.to(device)
        y = y.long().to(device)
        optimizer.zero_grad()
        # 获取teacher预测结果
        with torch.no_grad():
            teacher_pred = teacher(x)

        y_pred_no_mix = student(x)
        loss_ce, loss_dkd = dkd_loss(y_pred_no_mix, teacher_pred, y, alpha, beta, T)

        if not mixup_conf['enable']:
            loss = loss_ce + loss_dkd
        else:
            if np.random.rand(1) < mixup_conf['p']:
                if mixup_conf['cut']:
                    mel = student[0](x)
                    x_mix, y_a, y_b, lamb = cutmix(mel, y, mixup_conf['alpha'])
                    x_mix = x_mix.to(device)
                    y_a = y_a.to(device)
                    y_b = y_b.to(device)
                    y_pred_mix = student[1](x_mix)
                else:
                    x_mix, y_a, y_b, lamb = mixup(x, y, mixup_conf['alpha'])
                    x_mix = x_mix.to(device)
                    y_a = y_a.to(device)
                    y_b = y_b.to(device)
                    y_pred_mix = student(x_mix)
                loss_ce_a = F.cross_entropy(y_pred_mix, y_a)
                loss_ce_b = F.cross_entropy(y_pred_mix, y_b)
                loss_ce = lamb * loss_ce_a + (1 - lamb) * loss_ce_b
                loss = loss_ce + loss_dkd
            else:
                loss = loss_ce + loss_dkd

        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']

        with torch.no_grad():
            # 计算训练loss和acc（每批）
            y_ = torch.argmax(y_pred_no_mix, dim=1)
            correct += (y_ == y).sum().item()
            total += y.size(0)
            total_batch += 1
            sum_loss += loss.item()
            running_loss = sum_loss / total_batch
            running_acc = correct / total

        # 输出训练信息
        loop.set_description(f'Epoch [{start_epoch + epoch + 1}/{epochs + start_epoch}]')
        loop.set_postfix(lr=lr, loss=running_loss, acc=running_acc)

    scheduler.step()
    epoch_loss = sum_loss / total_batch
    epoch_acc = correct / total
    if save:
        torch.save(student.state_dict(),
                   "../model_weights/{}_epoch{}.pt".format(save_name, start_epoch + epoch + 1))  # 每轮保存一次参数
    return epoch_loss, epoch_acc


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
            * (temperature ** 2)
            / target.shape[0]
    )
    loss_ce = F.cross_entropy(logits_student, target)
    return loss_ce, alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    """
    用来获取目标标签对应的掩码
    """
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    """
    用来获取非目标标签对应的掩码
    """
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    """
    将预测结果与掩码拼接
    """
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
