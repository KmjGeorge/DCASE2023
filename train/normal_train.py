import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from optim.scheduler import GradualWarmupScheduler
import torch.nn as nn
from tqdm import tqdm
from dataset.augmentation import mixup
from train.trainingconfig import training_config
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



MAX_EPOCH = training_config['epoch']
TASK_NAME = training_config['task_name']
MIXUP_ALPHA = training_config['mixup_alpha']
CHOOSE_MODEL = training_config['model']


# 定义普通的训练过程，mixup_alpha为mixup参数，0不启用

def train(model, train_loader, test_loader, start_epoch, epochs, save_name, mixup_alpha, save=True):
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []

    criterion = nn.CrossEntropyLoss()
    # 先逐步增加至初始学习率，然后使用余弦退火
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=1e-7)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10,
                                              after_scheduler=scheduler_cos)
    scheduler_warmup.step()  # 学习率从0开始，跳过第一轮

    for i in range(epochs):
        epoch_loss, epoch_acc = train_per_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler_warmup,
            start_epoch=start_epoch,
            epoch=i,
            epochs=epochs,
            save_name=save_name,
            mixup_alpha=mixup_alpha,
            save=save)
        # 每轮验证一次
        val_epoch_loss, val_epoch_acc = validate(
            model=model,
            test_loader=test_loader,
            criterion=criterion)

        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        val_loss_list.append(val_epoch_loss)
        val_acc_list.append(val_epoch_acc)
        if save:
            logs = pd.DataFrame({'loss': epoch_loss,
                                 'acc': epoch_acc,
                                 'val_loss': val_epoch_loss,
                                 'val_acc': val_epoch_acc}
                                )
            logs.to_csv('../logs/{}_logs.csv'.format(TASK_NAME), index=True, mode='a')
    print('==========Finished Training===========')

    if save:
        torch.save(model, '../model_weights/{}_final.pt'.format(save_name))
    return loss_list, acc_list, val_loss_list, val_acc_list


def train_per_epoch(model, train_loader, criterion, optimizer, scheduler, start_epoch, epoch, epochs, save_name, mixup_alpha=0,
                    save=True):
    correct = 0
    total = 0
    sum_loss = 0.0
    model.train()

    loop = tqdm(train_loader)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        if mixup_alpha == 0:  # alpha=0 不使用mixup
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
        else:
            x_mix, y_a, y_b, lamb = mixup(x, y, mixup_alpha)
            x_mix = x_mix.to(device)
            y_a = y_a.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            y_pred = model(x_mix)
            # print('y_a.shape=', y_a.shape)
            # print('y_b.shape=', y_b.shape)
            # print('y_pred.shape=', y_pred.shape)
            loss = lamb * criterion(y_pred, y_a) + (1 - lamb) * criterion(y_pred, y_b)

        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']

        # 计算训练loss和acc（每批）
        with torch.no_grad():
            if mixup_alpha == 0:
                y_ = torch.argmax(y_pred, dim=1)
            else:
                y_ = torch.argmax(model(x), dim=1)  # 使用非mixup的数据计算acc
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
        torch.save(model.state_dict(), "../model_weights/{}_epoch{}.pt".format(save_name, start_epoch + epoch + 1))  # 每轮保存一次便于调试
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion):
    test_correct = 0
    test_total = 0
    test_sum_loss = 0
    model.eval()

    with torch.no_grad():
        loop = tqdm(test_loader, desc='Test')
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
