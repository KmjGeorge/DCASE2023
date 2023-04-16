import torch.optim
import optim.scheduler
import torch.nn as nn
normal_training_config = {
    'task_name': 'passt_tau2022_random_slicing_augment_mixup_mixstyle',   # 任务名，用于模型文件和日志命名
    'epoch': 250,
    'criterion': nn.CrossEntropyLoss(),
    'optim_config': {
        'name': torch.optim.AdamW,
        'lr': 1e-5,
        'weight_decay': 0.001,
    },
    'scheduler_cos_config': {
        'eta_min': 1e-7
    },
    'scheduler_warmup_config': {
        'multiplier': 1,
        'total_epoch': 30,
    },
    'model': 'passt',   # 目前可选:cp_resnet, mobileast_s, mobileast_xxs, rfr-cnn, passt, acdnet
}

