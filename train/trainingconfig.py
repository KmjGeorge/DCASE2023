training_config = {
    'mixup_alpha': 0.3,     # mixup超参数
    'task_name': 'mobileast_xxs_tau2022_random_slicing_augment_mixup',   # 任务名，用于模型文件和日志命名
    'epoch': 30,
    'model': 'cp_resnet'   # 目前可选:cp_resnet, mobileast_s, mobileast_xxs, rfr-cnn
}
