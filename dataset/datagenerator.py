import torch
import pandas as pd
import librosa
import librosa.feature
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from configs.dataconfig import spectrum_config, dataset_config
import h5py
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = dataset_config['name']
NUM_CLASSES = dataset_config['num_classes']
BATCH_SIZE = dataset_config['batch_size']
META_PATH = dataset_config['meta_path']
AUDIO_PATH = dataset_config['audio_path']
H5PATH = dataset_config['h5path']
SHUFFLE = dataset_config['shuffle']
SR = spectrum_config['sr']  # 采样率

TAU2022_CLASSES = {'airport': 0,
                   'bus': 1,
                   'metro': 2,
                   'metro_station': 3,
                   'park': 4,
                   'public_square': 5,
                   'shopping_mall': 6,
                   'street_pedestrian': 7,
                   'street_traffic': 8,
                   'tram': 9
                   }
TAU2022_DEVICES = {'a': 0,
                   'b': 1,
                   'c': 2,
                   's1': 3,
                   's2': 4,
                   's3': 5,
                   's4': 6,
                   's5': 7,
                   's6': 8
                   }

TAU2022_DEVICES_INVERT = {k: v for v, k in TAU2022_DEVICES.items()}

# 从h5读取TAU2022数据集
class TAU2022(Dataset):
    def __init__(self, h5_path, slicing=False):
        assert (
                DATASET_NAME == 'TAU2022' or DATASET_NAME == 'tau2022' or DATASET_NAME == 'TAU2022_RANDOM_SLICING' or DATASET_NAME == 'tau2022_random_slicing'
                or DATASET_NAME == 'TAU2022_REASSEMBLED' or DATASET_NAME == 'tau2022_reassembled')
        print('加载数据集...')
        with h5py.File(h5_path, 'r') as f:
            self.X = f['data'][:]
            self.Y = f['label'][:]
            self.Z = f['device'][:]
        print('加载完成')
        self.slicing = slicing

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, idx):
        audio = self.X[idx]
        label = self.Y[idx]
        device = self.Z[idx]
        if self.slicing:  # slicing 对10s版制作随机切片
            if len(audio) < SR * 10:
                raise '输入不是reassembled数据集！'
            clip_idx = random.randint(0, 9000)  # 总时长10000ms 取其中1000ms
            audio_clip = audio[int(clip_idx * SR / 1000): int((clip_idx + 1000) * SR / 1000)]
            return audio_clip, label, device
        else:
            return audio, label, device


# 构建tau2022数据集 以h5存储 包括：数据(波形) 类别标签 设备标签
def make_tau2022(meta_path, h5_path, reassembled=False):
    assert not os.path.exists(h5_path), "文件{}已存在！".format(h5_path)
    if reassembled:
        df = pd.read_csv(meta_path)
        df2 = pd.read_csv(dataset_config['meta_path'])
    else:
        df = pd.read_csv(meta_path, sep='\t')
        df2 = pd.read_csv(dataset_config['meta_path'], sep='\t')
    X = []
    Y = []
    Z = []
    loop = tqdm(df['filename'])
    for filename in loop:
        scene_label = df[df['filename'] == filename]['scene_label'].values[0]
        source_label = df2[df2['filename'] == filename]['source_label'].values[0]
        full_path = os.path.join(AUDIO_PATH, filename)
        audio, _ = librosa.load(full_path, sr=SR)
        X.append(audio)
        Y.append(TAU2022_CLASSES[scene_label])  # 标签转数字
        Z.append(TAU2022_DEVICES[source_label])
        loop.set_description('从audio文件读取数据集...')
    print('制作h5数据集中，可能需要数分钟时间...')
    with h5py.File(h5_path, 'a') as f:
        f.create_dataset('data', data=X)
        f.create_dataset('label', data=Y)
        f.create_dataset('device', data=Z)
    print('制作h5数据集完成，保存至 {}'.format(h5_path))


# 获取原版tau2022数据集
def get_tau2022():
    train_h5 = os.path.join(H5PATH, 'tau2022_train.h5')
    test_h5 = os.path.join(H5PATH, 'tau2022_test.h5')
    TAU2022_train = TAU2022(train_h5)
    TAU2022_test = TAU2022(test_h5)
    train = DataLoader(TAU2022_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test = DataLoader(TAU2022_test, batch_size=BATCH_SIZE, shuffle=False)
    return train, test


# 获得10s版数据集
def get_tau2022_reassembled():
    train_h5 = os.path.join(H5PATH, 'tau2022_reassembled_train.h5')
    test_h5 = os.path.join(H5PATH, 'tau2022_reassembled_test.h5')
    tau2022_train = TAU2022(train_h5)
    tau2022_test = TAU2022(test_h5)
    train = DataLoader(tau2022_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test = DataLoader(tau2022_test, batch_size=BATCH_SIZE, shuffle=False)
    return train, test


# 获取经过reassemble并随机切取1s的tau2022数据集
def get_tau2022_reassembled_random_slicing():
    train_h5 = os.path.join(H5PATH, 'tau2022_reassembled_train.h5')
    test_h5 = os.path.join(H5PATH, 'tau2022_test.h5')  # 测试集为1s原版
    tau2022_random_slicing_train = TAU2022(train_h5, slicing=True)
    tau2022_random_slicing_test = TAU2022(test_h5)
    train = DataLoader(tau2022_random_slicing_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test = DataLoader(tau2022_random_slicing_test, batch_size=BATCH_SIZE, shuffle=False)
    return train, test


if __name__ == '__main__':
    '''
    train_h5 = os.path.join(H5PATH, 'tau2022_train.h5')
    test_h5 = os.path.join(H5PATH, 'tau2022_test.h5')
    train_csv = os.path.split(META_PATH)[0] + '/evaluation_setup/fold1_train.csv'
    test_csv = os.path.split(META_PATH)[0] + '/evaluation_setup/fold1_evaluate.csv'
    make_tau2022(train_csv, train_h5, reassembled=False)
    make_tau2022(test_csv, test_h5, reassembled=False)
    '''

    # 原版TAU2022数据集调用
    TAU2022_train, TAU2022_test = get_tau2022()
    i = 0
    # 输出一批数据的shape
    for x, y, z in TAU2022_train:
        if i == 1:
            break
        print(x.shape)
        print(y.shape)
        print(z.shape)
        # print(x)
        # print(y)
        i += 1
    '''
    # TAU2022_reassembled数据集调用
    TAU2022_reassembled_train, TAU2022_reassembled_test = get_tau2022_reassembled()
    i = 0
    # 输出一批数据的shape
    from run.testscript import plot_waveform

    for x, y, z in TAU2022_reassembled_train:
        if i == 1:
            break
        plot_waveform(torch.unsqueeze(x[0], dim=0), sample_rate=32000)
        print(y[0])
        print(z[0])
        i += 1
    '''

    '''
    # random_slicing数据集调用
    Random_Slicing_train, Random_Slicing_test = get_tau2022_reassembled_random_slicing()
    i = 0
    # 输出一批数据的shape
    for x, y in Random_Slicing_train:
        if i == 1:
            break
        print(x.shape)
        print(y.shape)
        # print(x)
        # print(y)
        i += 1
    '''

    ''' urbansound8k 测试用
    Urbansound8k_train, UrbanSound8K_test = get_urbansound8k(fold_shuffle)
    i = 0
    for x, y in Urbansound8k_train:
        if i == 1:
            break
        print(x)
        print(y)
        print(x.shape)
        print(y.shape)
        i += 1
    '''
