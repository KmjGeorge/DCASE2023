import numpy as np
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import librosa


def plot_waveform(waveform, sample_rate):
    # waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle("Waveform")
    plt.show(block=False)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(specgram[0], origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


# librosa提取log mel
def extract_spectrum(audio, sr, n_fft, hop_length, win_length, n_mels):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    stft_abs = np.abs(stft) ** 2  # 取模
    mel = librosa.feature.melspectrogram(S=stft_abs, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)  # 取对数
    # 归一化
    mean = np.mean(log_mel, 0, keepdims=True)
    std = np.std(log_mel, 0, keepdims=True)
    log_mel = (log_mel - mean) / (std + 1e-5)
    log_mel = log_mel.astype('float32')
    log_mel = log_mel[np.newaxis, ...]  # (1, n_mels, sr*time//hop_length + 1)

    return log_mel


# torchaudio提取log mel + specaugment
class ExtractMel(nn.Module):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)

        self.win_length = kwargs['win_length']
        self.n_mels = kwargs['n_mels']
        self.n_fft = kwargs['n_fft']
        self.sr = kwargs['sr']
        self.fmin = kwargs['fmin']
        if kwargs['fmax'] is None:
            self.fmax = kwargs['sr'] // 2 - kwargs['fmax_aug_range'] // 2
            print(f"Warning: FMAX is None setting to {kwargs['fmax']} ")
        else:
            self.fmax = kwargs['fmax']
        self.hop_length = kwargs['hop_length']


        self.fmin_aug_range = kwargs['fmin_aug_range']
        self.fmax_aug_range = kwargs['fmax_aug_range']
        assert kwargs['fmin_aug_range'] >= 1, f"fmin_aug_range={self.fmin_aug_range} should be >=1; 1 means no augmentation"
        assert kwargs['fmax_aug_range'] >= 1, f"fmax_aug_range={self.fmax_aug_range} should be >=1; 1 means no augmentation"

        self.register_buffer('window',
                             torch.hann_window(kwargs['win_length'], periodic=False),  # Hann窗，窗长为win_length
                             persistent=False)
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)  # 预加重参数
        if kwargs['freqm'] == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(kwargs['freqm'], iid_masks=True)
        if kwargs['timem'] == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(kwargs['timem'], iid_masks=True)

    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)  # 预加重

        x = torch.stft(x, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,  # stft
                       center=True, normalized=False, window=self.window, return_complex=False)

        x = (x ** 2).sum(dim=-1)  # 获得幅度谱
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()

        # 只在训练时使用mel滤波器组的随机偏移
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, self.n_fft, self.sr,
                                                                 fmin, fmax, vtln_low=100.0, vtln_high=-500.,
                                                                 vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()  # 防止0

        if self.training:  # 只在训练时采用specaugment
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return torch.unsqueeze(melspec, dim=1)  # 添加一个通道维度


if __name__ == '__main__':
    from configs.dataconfig import spectrum_config
    # 测试
    extmel = ExtractMel(**spectrum_config)
    batch = []
    x1, _ = librosa.load('D:/Datasets/TAU-urban-acoustic-scenes-2022-mobile-development/audio/airport-barcelona-0-0-0-a.wav', sr=spectrum_config['sr'])
    x2, _ = librosa.load('D:/Datasets/TAU-urban-acoustic-scenes-2022-mobile-development/audio/airport-barcelona-0-0-1-a.wav', sr=spectrum_config['sr'])
    print(x1.shape)
    plot_waveform(x1[np.newaxis, ...], sample_rate=spectrum_config['sr'])
    batch.append(x1)
    batch.append(x2)
    batch = torch.tensor(batch)
    print(batch.shape)  # (batchsize, sr*time)
    batch = extmel(batch)
    print(batch.shape)    # (batchsize, C, F, T)  其中C=1, F=n_mels, T=sr*time // hop_length 向上取整
    plot_spectrogram(batch[0])

