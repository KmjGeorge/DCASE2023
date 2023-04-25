import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim import AdamW
from torch.optim.sgd import SGD
import matplotlib.pyplot as plt
from scheduler import GradualWarmupScheduler, ExpWarmupLinearDownScheduler
from model_src.rfr_cnn import RFR_CNN

MAX_EPOCH = 200
BATCH_SIZE = 32
if __name__ == '__main__':

    model = RFR_CNN()

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)

    # optimizer = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = StepLR(optimizer, step_size=5, gamma=0.1)

    # 先逐步增加至初始学习率，然后使用余弦退火
    # scheduler_cos = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH-10, eta_min=1e-5)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler_cos)
    scheduler_exp_warmup_linear_down = ExpWarmupLinearDownScheduler(optimizer, warm_up_len=50, ramp_down_start=50,
                                                                    ramp_down_len=100, last_lr_value=5e-3)
    # optimizer.zero_grad()
    # optimizer.step()
    lr_list = []
    for epoch in range(MAX_EPOCH):
        for batch in range(BATCH_SIZE):
            optimizer.zero_grad()
            optimizer.step()
        scheduler_exp_warmup_linear_down.step()
        cur_lr = optimizer.param_groups[0]['lr']
        lr_list.append(cur_lr)
        print(epoch + 1, cur_lr)

    plt.plot(lr_list)
    plt.show()
