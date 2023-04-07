import torch
from dataset.datagenerator import get_urbansound8k
from training_config import training_config
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from optim.scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from size_cal import nessi
from model_src.mobilevit import mobileast_xxs, mobileast_s
from model_src.teacher.rfr_cnn import RFR_CNN
from model_src.student.cp_resnet import cp_resnet
from dataset.augmentation import mixup
from dataset.spectrum import ExtractMel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_EPOCH = training_config['epoch']
TASK_NAME = training_config['task_name']
MIXUP_ALPHA = training_config['mixup_alpha']
CHOOSE_MODEL = training_config['model']


def train(model, train_loader, test_loader, epochs, save_name, mixup_alpha):
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []

    criterion = nn.CrossEntropyLoss()
    # 先逐步增加至初始学习率，然后使用余弦退火
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=1e-4)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3,
                                              after_scheduler=scheduler_cos)
    for i in range(epochs):
        epoch_loss, epoch_acc = train_per_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler_warmup,
            epoch=i,
            epochs=epochs,
            save_name=save_name,
            mixup_alpha=mixup_alpha)
        # 每轮验证一次
        test_epoch_loss, test_epoch_acc = test_per_epoch(
            model=model,
            test_loader=test_loader,
            criterion=criterion)

        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        val_loss_list.append(test_epoch_loss)
        val_acc_list.append(test_epoch_acc)
    print('==========Finished Training===========')
    torch.save(model, '../model_weights/{}.pt'.format(save_name))
    print('=========Finished Saving============')
    return loss_list, acc_list, val_loss_list, val_acc_list


def train_per_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, epochs, save_name, mixup_alpha=0):
    correct = 0
    total = 0
    sum_loss = 0.0
    model.train()

    loop = tqdm(train_loader)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        if mixup_alpha == 0:  # 不使用mixup
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
                y_ = torch.argmax(model(x), dim=1)  # 使用正常数据进行预测
            correct += (y_ == y).sum().item()
            total += y.size(0)
            sum_loss += loss.item()
            running_loss = sum_loss / total
            running_acc = correct / total

        # 输出训练信息
        loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
        loop.set_postfix(lr=lr, loss=running_loss, acc=running_acc)
    scheduler.step()  # 更新学习率
    epoch_loss = sum_loss / total
    epoch_acc = correct / total
    torch.save(model.state_dict(), "../model_weights/{}_epoch{}.pt".format(save_name, epoch + 1))
    return epoch_loss, epoch_acc


def test_per_epoch(model, test_loader, criterion):
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

            # 输出测试信息
            loop.set_postfix(val_loss=test_running_loss, val_acc=test_running_acc)

    test_epoch_loss = test_sum_loss / test_total
    test_epoch_acc = test_correct / test_total
    model.train()

    return test_epoch_loss, test_epoch_acc


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


if __name__ == '__main__':
    # audio
    from dataset.dataconfig import spectrum_config

    if CHOOSE_MODEL == 'cp_resnet':
        model = nn.Sequential(ExtractMel(**spectrum_config), cp_resnet()).to(device)
    elif CHOOSE_MODEL == 'mobileast_s':
        model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_s()).to(device)
    elif CHOOSE_MODEL == 'mobileast_xxs':
        model = nn.Sequential(ExtractMel(**spectrum_config), mobileast_xxs()).to(device)
    elif CHOOSE_MODEL == 'RFR-CNN':
        model = nn.Sequential(ExtractMel(**spectrum_config), RFR_CNN().to(device))
    else:
        raise 'Undefied Model'

    nessi.get_model_size(model, 'torch', input_size=(1, spectrum_config['sr'] * 4))
    train_dataloader, test_dataloader = get_urbansound8k(fold_shuffle=True)

    # image
    '''
    model = mobilevit_s().to(device)
    summary(model, input_size=(3, 256, 256))

    train_dataloader, test_dataloader = get_imagenetsub(1)
    '''

    loss_list, acc_list, val_loss_list, val_acc_list = train(model, train_dataloader, test_dataloader,
                                                             epochs=MAX_EPOCH, save_name=TASK_NAME,
                                                             mixup_alpha=MIXUP_ALPHA)
    logs = pd.DataFrame({'loss': loss_list,
                         'acc': acc_list,
                         'val_loss': val_loss_list,
                         'val_acc': val_acc_list}
                        )
    logs.to_csv('../logs/{}_logs.csv'.format(TASK_NAME), index_label=True)
    show_accloss(loss_list, acc_list, val_loss_list, val_acc_list, name=TASK_NAME)
