spectrum_config = {
    'sr': 16000,
    'n_fft': 1024,
    'win_length': 1000,
    'hop_length': 500,
    'n_mels': 128,
    'freqm': 20,
    'timem': 20,
    'fmin': 0.0,
    'fmax': None,
    'fmin_aug_range': 1,
    'fmax_aug_range': 1000
}

dataset_config = {
    'name': 'tau2022_random_slicing',
    'batch_size': 64,
    'meta_path': 'G:/datasets/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv',
    'audio_path': 'G:/datasets/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/',   # 如果是TAU2022, 路径不进到audio文件夹
    'h5_path': './h5/tau2022_reassemble.h5',  # 当使用reassembled数据集时启用该项，否则为None
    'num_classes': 10,
    'shuffle': True
}




