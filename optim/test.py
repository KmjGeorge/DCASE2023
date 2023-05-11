import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim import AdamW
from torch.optim.sgd import SGD
import matplotlib.pyplot as plt
from scheduler import GradualWarmupScheduler, ExpWarmupLinearDownScheduler, ExpWarmupCosineDownScheduler
from model_src.quant_mobilevit import mobileast_light2
MAX_EPOCH = 500
BATCH_SIZE = 32
if __name__ == '__main__':

    model = mobileast_light2(mixstyle_conf={'enable': False})

    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.001)

    # optimizer = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = StepLR(optimizer, step_size=5, gamma=0.1)

    # 先逐步增加至初始学习率，然后使用余弦退火
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH-30, eta_min=1e-5)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=30, after_scheduler=scheduler_cos)
    # scheduler_exp_warmup_linear_down = ExpWarmupLinearDownScheduler(optimizer, warm_up_len=50, ramp_down_start=50,
    #                                                                 ramp_down_len=200, last_lr_value=0.1)
    # scheduler_exp_warmup_cosine_down = ExpWarmupCosineDownScheduler(optimizer, warmup=30, rampdown_length=10, num_epochs=500)
    # optimizer.zero_grad()
    # optimizer.step()
    lr_list = []
    for epoch in range(MAX_EPOCH):
        for batch in range(BATCH_SIZE):
            optimizer.zero_grad()
            optimizer.step()
        scheduler_warmup.step()
        # scheduler_exp_warmup_linear_down.step()
        # scheduler_exp_warmup_cosine_down.step()
        cur_lr = optimizer.param_groups[0]['lr']
        lr_list.append(cur_lr)
        print(epoch + 1, cur_lr)

    plt.plot(lr_list)
    plt.show()
