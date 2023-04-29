import torch.optim
import optim.scheduler
import torch.nn as nn

normal_training_config = {
    'task_name': 'mobileastv3_Light2_mixstyle(alpha=0.3, p=0.6), T=1, soft_loss_alpha=50_logs_(lr=1e-3 30, 5e-2 150)',  # 任务名，用于模型文件和日志命名
    'epoch': 300,
    'start_epoch': 0,
    'criterion': nn.CrossEntropyLoss(),
    'optim_config': {
        'name': torch.optim.Adam,
        'lr': 1e-3,
        'weight_decay': 0.001,
    },
    'scheduler_warmup_config': {
        'multiplier': 1,
        'total_epoch': 30,
    },
    'scheduler_down_config': {
        'total_epoch': 150,
        'eta_min': 5e-2,
    },
    'model': 'mobileast_light2',
}

distillation_config = {
    'task_name': 'passt+mobileastv3_Light2 T=1, soft_loss_alpha=50',
    # 任务名，用于模型文件和日志命名
    'epoch': 500,
    'start_epoch': 0,
    'hard_criterion': nn.CrossEntropyLoss(),
    'soft_criterion': nn.KLDivLoss(reduction='batchmean', log_target=False),
    'optim_config': {
        'name': torch.optim.Adam,
        'lr': 1e-3,
        'weight_decay': 0.001,
    },
    'scheduler_warmup_config': {
        'multiplier': 1,
        'total_epoch': 30,
    },
    'scheduler_down_config': {
        'total_epoch': 250,
        'eta_min': 5e-2,
    },
    'teacher_model': 'passt',
    'teacher_weight_path': '../model_weights/passt_tau2022_random_slicing_augment_fpatchout=6_mixup(alpha=0.3)_mixstyle(alpha=0.3,p=0.6)_valacc=59.87.pt',
    'student_model': 'mobileast_light2',
    'T': 1,  # 蒸馏温度
    'alpha': 50,  # soft_loss损失系数

    'loss_device_weight': 1
}

