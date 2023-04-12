import torch
import pandas as pd
import librosa
import librosa.feature
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.dataconfig import spectrum_config, dataset_config
import h5py
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = dataset_config['name']
NUM_CLASSES = dataset_config['num_classes']
BATCH_SIZE = dataset_config['batch_size']
META_PATH = dataset_config['meta_path']
AUDIO_PATH = dataset_config['audio_path']
REASSEMBLED_AUDIO_PATH = dataset_config['reassembled_audio_path']
REASSEMBLED_META_PATH = dataset_config['reassembled_meta_path']
H5PATH = dataset_config['h5path']
SLICING_H5PATH = dataset_config['slicing_h5path']
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


# 从文件读取urbansound8k数据集
class UrbanSound8K(Dataset):
    def __init__(self, meta_csv_path, fold_list):
        assert (DATASET_NAME == 'urbansound8k' or DATASET_NAME == 'URBANSOUND8K')
        self.X = []
        self.Y = []
        df = pd.read_csv(meta_csv_path)
        loop = tqdm(df['slice_file_name'])
        for filename in loop:
            fold_num = df[df['slice_file_name'] == filename]['fold'].values[0]
            if fold_num not in fold_list:
                continue
            fold = 'fold' + str(fold_num) + '/'
            full_path = os.path.join(AUDIO_PATH, fold, filename)

            time = librosa.get_duration(path=full_path)
            if time >= 4.0:
                classID = df[df['slice_file_name'] == filename]['classID'].values[0]
                self.X.append(full_path)
                self.Y.append(classID)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.X[idx], sr=SR)
        if audio.shape[0] != 64000:  # 超出4s的裁剪为4s
            audio = audio[:64000]

        label = self.Y[idx]
        return audio, label


# 从h5读取TAU2022数据集
class TAU2022(Dataset):
    def __init__(self, h5_path):
        assert (
                DATASET_NAME == 'TAU2022' or DATASET_NAME == 'tau2022' or DATASET_NAME == 'TAU2022_RANDOM_SLICING' or DATASET_NAME == 'tau2022_random_slicing')

        self.X = []
        Y = []
        with h5py.File(h5_path, 'r') as f:
            loop = tqdm(f.keys())
            for k in loop:
                label = k
                for _, dataset in f[label].items():
                    for data in dataset:
                        # print('data=', data)
                        # print(data.shape)
                        self.X.append(data)
                        Y.append(TAU2022_CLASSES[label])  # 标签转数字
                loop.set_description('从h5加载tau2022数据集...')
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        audio = self.X[idx]
        label = self.Y[idx]
        return audio, label


# 构建tau2022数据集 以h5存储
def make_tau2022(meta_path, h5_path):
    assert not os.path.exists(h5_path), "文件{}已存在！".format(h5_path)
    df = pd.read_csv(meta_path, sep='\t')
    X = []
    Y = []
    loop = tqdm(df['filename'])
    for filename in loop:
        scene_label = df[df['filename'] == filename]['scene_label'].values[0]
        full_path = os.path.join(AUDIO_PATH, filename)
        audio, _ = librosa.load(full_path, sr=SR)
        X.append(audio)
        Y.append(scene_label)
        loop.set_description('从audio文件读取tau2022数据集...')

    loop2 = tqdm(range(len(X)))
    for i in loop2:
        data = np.array(X[i])
        label = Y[i]
        with h5py.File(h5_path, 'a') as f:
            try:
                label_group = f[label]
            except:
                label_group = f.create_group(label)
            try:
                dataset = label_group['x']
                dataset.resize((dataset.shape[0] + 1, dataset.shape[1]))
                dataset[-1] = data
            except:
                label_group.create_dataset('x', data=[data], maxshape=(None, SR), chunks=True)
        loop2.set_description('制作tau2022 h5数据集...')
    print('数据保存至 {}'.format(h5_path))


# 从tau2022_reassembled 10s的音频中随机取1s  n次，构成新数据集，以h5存储
def make_tau2022_reassemble_random_slicing(n):
    assert not os.path.exists(SLICING_H5PATH), "文件{}已存在！".format(SLICING_H5PATH)
    train_path = os.path.split(REASSEMBLED_META_PATH)[0] + '/evaluation_setup/fold1_train.csv'
    df = pd.read_csv(train_path)
    X = []
    Y = []
    loop = tqdm(df['filename'])
    for filename in loop:
        scene_label = df[df['filename'] == filename]['scene_label'].values[0]
        full_path = os.path.join(REASSEMBLED_AUDIO_PATH, filename)
        audio, _ = librosa.load(full_path, sr=SR)
        X.append(audio)
        Y.append(scene_label)
        loop.set_description('从audio文件读取reassembled数据集...')

    loop2 = tqdm(range(len(X)))
    for i in loop2:
        data = np.array(X[i])
        label = Y[i]
        with h5py.File(SLICING_H5PATH, 'a') as f:
            try:
                label_group = f[label]
            except:
                label_group = f.create_group(label)
            for i in range(n):
                clip_idx = random.randint(0, 9000)  # 总时长10000ms 随机取其中1000ms
                x_new = data[int(clip_idx * SR / 1000): int((clip_idx + 1000) * SR / 1000)]
                try:
                    dataset = label_group['x']
                    dataset.resize((dataset.shape[0] + 1, dataset.shape[1]))
                    dataset[-1] = x_new
                except:
                    label_group.create_dataset('x', data=[x_new], maxshape=(None, SR), chunks=True)
        loop2.set_description('正在进行随机切片...')
    print('random_slicing制作完成，数据保存至 {}'.format(SLICING_H5PATH))


# reassemble数据集随机切取1s得到的数据集，从h5读取
class TAU2022_Random_Slicing(Dataset):
    def __init__(self, h5_path):
        assert (DATASET_NAME == 'TAU2022_RANDOM_SLICING' or DATASET_NAME == 'tau2022_random_slicing')
        self.X = []
        Y = []
        with h5py.File(h5_path, 'r') as f:
            loop = tqdm(f.keys())
            for k in loop:
                label = k
                for _, dataset in f[label].items():
                    for data in dataset:
                        # print('data=', data)
                        # print(data.shape)
                        self.X.append(data)
                        Y.append(TAU2022_CLASSES[label])  # 标签转数字
                loop.set_description('从h5加载random_slicing数据集...')
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        audio = self.X[idx]
        label = self.Y[idx]
        return audio, label


# 制作数据集，返回波形，选择9个fold为训练集，1个为验证集
def get_urbansound8k(fold_shuffle):
    if fold_shuffle:
        test_fold = random.randint(1, 10)
        train_fold = [i for i in range(1, 10) if i != test_fold]
        test_fold = [test_fold]
    else:
        train_fold = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_fold = [10]
    print('train fold : ', train_fold, 'test fold :', test_fold)
    UrbanSound8K_train = UrbanSound8K(AUDIO_PATH, fold_list=train_fold)
    UrbanSound8K_test = UrbanSound8K(AUDIO_PATH, fold_list=test_fold)
    train = DataLoader(UrbanSound8K_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test = DataLoader(UrbanSound8K_test, batch_size=BATCH_SIZE, shuffle=False)
    return train, test


# 获取原版tau2022数据集
def get_tau2022():
    train_h5 = os.path.join(H5PATH, 'tau2022_train.h5')
    test_h5 = os.path.join(H5PATH, 'tau2022_test.h5')
    TAU2022_train = TAU2022(train_h5)
    # print('len(TAU2022_train)=', len(TAU2022_train))  # 139620
    TAU2022_test = TAU2022(test_h5)
    # print('len(TAU2022_test)=', len(TAU2022_test))  # 29680
    train = DataLoader(TAU2022_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test = DataLoader(TAU2022_test, batch_size=BATCH_SIZE, shuffle=False)
    return train, test


# 获取经过reassemble并随机切取1s的tau2022数据集
def get_tau2022_random_slicing():
    train_h5 = SLICING_H5PATH
    test_h5 = os.path.join(H5PATH, 'tau2022_test.h5')
    TAU2022_random_slicing_train = TAU2022_Random_Slicing(train_h5)
    # print('len(slicing_train)=', len(TAU2022_random_slicing_train))  # 139620
    TAU2022_test = TAU2022(test_h5)
    # print('len(slicing_test)=', len(TAU2022_test))  # 29680
    train = DataLoader(TAU2022_random_slicing_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test = DataLoader(TAU2022_test, batch_size=BATCH_SIZE, shuffle=False)  # 测试集
    return train, test


if __name__ == '__main__':
    # 以H5制作TAU2022原版数据集，存放到dataconfig['h5_ath']
    train_h5 = os.path.join(H5PATH, 'tau2022_train.h5')
    test_h5 = os.path.join(H5PATH, 'tau2022_test.h5')
    train_csv = os.path.split(META_PATH)[0] + '/evaluation_setup/fold1_train.csv'
    test_csv = os.path.split(META_PATH)[0] + '/evaluation_setup/fold1_evaluate.csv'
    make_tau2022(train_csv, train_h5)
    make_tau2022(test_csv, test_h5)

    # 以H5制作根据reassemble的数据集生成10s随机采样为若干1s的数据集，存放到dataconfig['slicing_h5_path']
    # 在此之前，请先运行dataset/files_reassemble.ipynb和dataset/meta_csv_reassemble.ipynb 以制作reassemble数据集，
    make_tau2022_reassemble_random_slicing(n=10)  # 参数n为随机采样的次数，最后得到的数据集为原数据集的n倍

    '''原版TAU2022数据集调用
    TAU2022_train, TAU2022_test = get_tau2022()
    i = 0
    # 输出一批数据的shape
    for x, y in TAU2022_train:
        if i == 1:
            break
        print(x.shape)
        print(y.shape)
        # print(x)
        # print(y)
        i += 1
    '''

    '''random_slicing数据集调用
    Random_Slicing_train, Random_Slicing_test = get_tau2022_random_slicing()
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
