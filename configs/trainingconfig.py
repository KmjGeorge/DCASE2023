import torch.optim
import optim.scheduler
import torch.nn as nn

normal_training_config = {
    'task_name': 'mobileast_cpresnet',  # 任务名，用于模型文件和日志命名
    'epoch': 200,
    'start_epoch': 0,
    'criterion': nn.CrossEntropyLoss(),
    'optim_config': {
        'name': torch.optim.Adam,
        'lr': 1e-3,
        'weight_decay': 0.001,
    },
    'scheduler_warmup_config': {
        'multiplier': 1,
        'total_epoch': 10,
    },
    'scheduler_down_config': {
        'total_epoch': 100,
        'eta_min': 5e-3,
    },
    'model': 'mobileast_light',  # 目前可选:cp_resnet, mobileast_s, mobileast_xxs, mobileast_light, rfr-cnn, passt, acdnet
}

distillation_config = {
    'task_name': 'passt+mobileast_light_cpresnet_tau2022_random_slicing_cutmix(alpha=0.3)_mixstyle(alpha=0.3, p=0.8), T=2, soft_loss_alpha=50',
    # 任务名，用于模型文件和日志命名
    'epoch': 200,
    'start_epoch': 0,
    'hard_criterion': nn.CrossEntropyLoss(),
    'soft_criterion': nn.KLDivLoss(reduction='batchmean', log_target=False),
    'optim_config': {
        'name': torch.optim.AdamW,
        'lr': 5e-4,
        'weight_decay': 0.01,
    },
    'scheduler_cos_config': {
        'eta_min': 1e-6
    },
    'scheduler_warmup_config': {
        'multiplier': 1,
        'total_epoch': 10,
    },
    'teacher_model': 'passt',
    'teacher_weight_path': '../model_weights/passt_tau2022_random_slicing_augment_fpatchout=6_mixup(alpha=0.3)_mixstyle(alpha=0.3,p=0.6)_valacc=59.87.pt',
    'student_model': 'mobileast_light',
    'T': 1,  # 蒸馏温度
    'alpha': 50,  # soft_loss损失系数

    'loss_device_weight': 1
}

