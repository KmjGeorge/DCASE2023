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
    'batch_size': 64,
    'meta_path': 'D:/datasets/UrbanSound8K/metadata/UrbanSound8K.csv',
    'audio_path': 'D:/datasets/UrbanSound8K/audio/',
    'num_classes': 10
}




