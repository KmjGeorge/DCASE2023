import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim import AdamW
from torch.optim.sgd import SGD
import matplotlib.pyplot as plt
from scheduler import GradualWarmupScheduler

MAX_EPOCH = 30
BATCH_SIZE = 4
if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = AdamW(model, lr=1e-4, weight_decay=0.01)
    # optimizer = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=1e-5)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=4, after_scheduler=scheduler_cos)

    optimizer.zero_grad()
    optimizer.step()

    lr_list = []
    for epoch in range(1, MAX_EPOCH):
        for batch in range(1, BATCH_SIZE):
            optimizer.step()  # backward pass (update network)
        scheduler_warmup.step()
        cur_lr = optimizer.param_groups[0]['lr']
        lr_list.append(cur_lr)
        print(epoch, cur_lr)

    plt.plot(lr_list)
    plt.show()
