import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from optim.scheduler import GradualWarmupScheduler
from tqdm import tqdm
from train.trainingconfig import training_config
from train.normal_train import validate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_EPOCH = training_config['epoch']
TASK_NAME = training_config['task_name']
MIXUP_ALPHA = training_config['mixup_alpha']
CHOOSE_MODEL = training_config['model']


def train_student(student, teacher, train_loader, test_loader, epochs, save_name, save=True):
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []

    criterion = nn.CrossEntropyLoss()
    # 先逐步增加至初始学习率，然后使用余弦退火
    optimizer = optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=1e-5)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=int(MAX_EPOCH / 10) + 1,
                                              after_scheduler=scheduler_cos)
    for i in range(epochs):
        epoch_loss, epoch_acc = distillation(
            model=student,
            teacher=teacher,
            train_loader=train_loader,
            T=5,
            alpha=0.3,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler_warmup,
            epoch=i,
            epochs=epochs,
            save_name=TASK_NAME,
            save=True)
        # 每轮验证一次
        test_epoch_loss, test_epoch_acc = validate(
            model=student,
            test_loader=test_loader,
            criterion=criterion)

        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        val_loss_list.append(test_epoch_loss)
        val_acc_list.append(test_epoch_acc)
    print('==========Finished Training===========')
    if save:
        torch.save(student, '../model_weights/{}_final.pt'.format(save_name))
        print('=========Finished Saving============')
    return loss_list, acc_list, val_loss_list, val_acc_list


def distillation(model, teacher, train_loader, T, alpha, criterion, optimizer, scheduler, epoch, epochs, save_name, save):
    """
    T: 蒸馏温度
    alpha: 损失系数
    """
    correct = 0
    total = 0
    sum_loss = 0.0
    soft_criterion = nn.KLDivLoss(reduction='batchmean')
    model.train()

    loop = tqdm(train_loader)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)
        # 获取teacher预测结果
        with torch.no_grad():
            teacher_pred = teacher(x)

        # hard loss
        hard_loss = criterion(y_pred, y)  # nn.CrossEntropyLoss()的输入不需要softmax
        # soft loss
        soft_loss = soft_criterion(F.softmax(y_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1))
        loss = alpha * hard_loss + (1 - alpha) * soft_loss

        # 反向传播 更新优化器
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # 计算训练loss和acc（每批）
            y_ = torch.argmax(y_pred, dim=1)
            correct += (y_ == y).sum().item()
            total += y.size(0)
            sum_loss += hard_loss.item()
            running_loss = sum_loss / total
            running_acc = correct / total

        # 输出训练信息
        loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
        loop.set_postfix(loss=running_loss, acc=running_acc)

    scheduler.step()
    epoch_loss = sum_loss / total
    epoch_acc = correct / total
    if save:
        torch.save(model.state_dict(), "../model_weights/{}_epoch{}.pt".format(save_name, epoch + 1))  # 每轮保存一次便于调试
    return epoch_loss, epoch_acc
