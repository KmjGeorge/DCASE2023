import random

import torch
import numpy as np


# 对batch执行mixup，返回新的batch
# 注意mixup的实现还有一部分在训练阶段，见train.normal_train.train_per_epoch
def mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha)  # 采样λ
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)  # 随机选取混合下标
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# 添加高斯噪声，输入是一个波形, snr为信噪比(dB)
def gauss_noise(audio, snr):
    signal_power = np.sum(audio ** 2) / len(audio)    # 计算信号功率
    noise_power = signal_power / (10 ** (snr / 10))   # 计算噪声功率
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(audio))  # 噪声功率作为平方根作为高斯噪声的标准差
    return audio + noise


if __name__ == '__main__':
    import librosa
    from dataset.spectrum import plot_waveform

    audio, _ = librosa.load(
        'G:/datasets/TAU-urban-acoustic-scenes-2022-mobile-development/audio/airport-barcelona-0-0-0-a.wav', sr=32000)
    print(audio.shape)
    plot_waveform(audio[np.newaxis, ...], sample_rate=32000)
    audio_noise = gauss_noise(audio, snr=20)
    plot_waveform(audio_noise[np.newaxis, ...], sample_rate=32000)
