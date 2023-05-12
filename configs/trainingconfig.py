import torch.optim
import optim.scheduler
import torch.nn as nn

normal_training_config = {
    'task_name': 'passt mixup(0.3) mixstyle(0.3 0.6), (lr=1e-3 30, 1e-2 400)',
    # 任务名，用于模型文件和日志命名
    'epoch': 500,
    'start_epoch': 0,
    'criterion': nn.CrossEntropyLoss(reduction='mean'),
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
        'total_epoch': 100,
        'eta_min': 1e-2,
    },
    'model': 'damped_cp_resnet',
}

distillation_config = {
    'task_name': 'passt+mobileastv3_light2 nomvit2 SiLU fc=160 DKD T=4 alpha=1 beta=10 MixStyle(0.3 0.6) Mixup(0.3 1) 1e-3(30) 1e-2(400)',
    # 任务名，用于模型文件和日志命名
    'epoch': 500,
    'start_epoch': 0,
    'hard_criterion': nn.CrossEntropyLoss(reduction='mean'),
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
        'total_epoch': 400,
        'eta_min': 1e-2,
    },
    'teacher_model': 'passt',
    'teacher_weight_path': '../model_weights/best/passt_tau2022_random_slicing_augment_fpatchout=6_mixup(alpha=0.3)_mixstyle(alpha=0.3,p=0.6)_59.87.pt',
    'student_model': 'mobileast_light2_small',
    'T': 4,  # 蒸馏温度
    'alpha': 1,  # soft_loss损失系数（普通）， TCKD损失系数（DKD）
    'beta': 10,  # NCKD损失系数
    'loss_device_weight': 1
}
