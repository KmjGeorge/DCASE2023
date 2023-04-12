import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim import AdamW
from torch.optim.sgd import SGD
import matplotlib.pyplot as plt
from scheduler import GradualWarmupScheduler
from model_src.rfr_cnn import RFR_CNN

MAX_EPOCH = 30
BATCH_SIZE = 32
if __name__ == '__main__':

    model = RFR_CNN()

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # optimizer = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = StepLR(optimizer, step_size=5, gamma=0.1)

    # 先逐步增加至初始学习率，然后使用余弦退火
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=1e-5)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cos)
    # optimizer.zero_grad()
    # optimizer.step()
    lr_list = []
    for epoch in range(MAX_EPOCH):
        for batch in range(BATCH_SIZE):
            optimizer.zero_grad()
            optimizer.step()
        scheduler_warmup.step()
        cur_lr = optimizer.param_groups[0]['lr']
        lr_list.append(cur_lr)
        print(epoch + 1, cur_lr)

    plt.plot(lr_list)
    plt.show()
