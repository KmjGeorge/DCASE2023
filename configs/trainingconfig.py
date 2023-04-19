import torch.optim
import optim.scheduler
import torch.nn as nn
'''
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
'''

normal_training_config = {
    'task_name': 'cp_resnet_tau2022_random_silcing_mixup_mixstyle',   # 任务名，用于模型文件和日志命名
    'epoch': 100,
    'criterion': nn.CrossEntropyLoss(),
    'optim_config': {
        'name': torch.optim.AdamW,
        'lr': 1e-3,
        'weight_decay': 0.01,
    },
    'scheduler_cos_config': {
        'eta_min': 1e-5
    },
    'scheduler_warmup_config': {
        'multiplier': 1,
        'total_epoch': 10,
    },
    'model': 'cp_resnet',   # 目前可选:cp_resnet, mobileast_s, mobileast_xxs, mobileast_light, rfr-cnn, passt, acdnet
}


distillation_config = {
    'task_name': 'passt+cp_resnet_tau2022_random_slicing_mixup_mixstyle',   # 任务名，用于模型文件和日志命名
    'epoch': 150,
    'hard_criterion': nn.CrossEntropyLoss(),
    'soft_criterion': nn.KLDivLoss(reduction='batchmean', log_target=False),
    'optim_config': {
        'name': torch.optim.AdamW,
        'lr': 1e-3,
        'weight_decay': 0.01,
    },
    'scheduler_cos_config': {
        'eta_min': 1e-5
    },
    'scheduler_warmup_config': {
        'multiplier': 1,
        'total_epoch': 10,
    },
    'teacher_model': 'passt',
    'teacher_weight_path': '../model_weights/passt_tau2022_random_slicing_augment_fpatchout=6_mixup(alpha=0.3)_mixstyle(alpha=0.3,p=0.6)_valacc=59.87.pt',
    'student_model': 'cp_resnet',
    'T': 2.5,      # 蒸馏温度
    'alpha': 0.5  # 损失系数
}
