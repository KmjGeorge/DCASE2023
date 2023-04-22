import random

import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 对batch执行mixup，返回新的batch
# 注意mixup的实现还有一部分在训练阶段，见train.normal_train.train_per_epoch
def mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha)  # 采样λ
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)  # 随机选取混合下标
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# 添加高斯噪声，输入是一个batch的波形, snr为信噪比(dB)
def gauss_noise(batch_audio, snr):
    noise_batch = []
    for audio in batch_audio:
        audio_numpy = audio.cpu().numpy()
        signal_power = np.sum(audio_numpy ** 2) / len(audio_numpy)  # 计算信号功率
        noise_power = signal_power / (10 ** (snr / 10))  # 计算噪声功率
        noise = np.random.normal(scale=np.sqrt(noise_power), size=len(audio_numpy))  # 噪声功率作为平方根作为高斯噪声的标准差
        noise_batch.append(audio_numpy + noise)
    noise_batch = torch.tensor(noise_batch, dtype=torch.float32).to(device)
    # print(noise_batch.shape)
    return noise_batch


def cutmix(data, target, alpha):
    """
    CutMix augmentation implementation.
    参数:
        data: batch of input images, shape (N, C, H, W)
        target: batch of target vectors, shape (N,)
        alpha: hyperparameter controlling the strength of CutMix regularization
    Returns:
        data: batch of mixed images, shape (N, C, H, W)
        target_a: batch of target vectors type A, shape (N,)
        target_b: batch of target vectors type B, shape (N,)
        lam: Mixing ratio of types A and B
    """
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    # Compute output
    target_a = target
    target_b = shuffled_target
    return data, target_a, target_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


if __name__ == '__main__':
    import librosa
    from dataset.spectrum import plot_waveform

    audio, _ = librosa.load(
        'G:/datasets/TAU-urban-acoustic-scenes-2022-mobile-development/audio/airport-barcelona-0-0-0-a.wav', sr=32000)
    print(audio.shape)
    plot_waveform(audio[np.newaxis, ...], sample_rate=32000)
    audio_noise = gauss_noise(audio, snr=20)
    plot_waveform(audio_noise[np.newaxis, ...], sample_rate=32000)
