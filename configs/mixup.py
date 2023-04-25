mixup_config = {
    'enable': False,
    'alpha': 0.3,
    'p': 1.0,
    'cut': False  # 使用cutmix时，网络输入应是频谱图而不是波形 在数据集中添加mel=True，并在网络中移除ExtractMel层
}
