training_config = {
    'mixup_alpha': 0.3,     # mixup超参数
    'task_name': 'passt_tau2022_augment_mixup_mixstyle',   # 任务名，用于模型文件和日志命名
    'epoch': 250,
    'model': 'passt',   # 目前可选:cp_resnet, mobileast_s, mobileast_xxs, rfr-cnn, passt, acdnet
    'mixstyle': {
            'enable': True,
            'alpha': 0.3,
            'p': 0.6,
            'freq': True
    }  # mixstyle参数： 启用与否，alpha，p，是否在频率上启用
}
