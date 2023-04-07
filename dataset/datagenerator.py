import random
import torch
import pandas as pd
import torchaudio
import librosa
import librosa.feature
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.dataconfig import spectrum_config, dataset_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

NUM_CLASSES = dataset_config['num_classes']
BATCH_SIZE = dataset_config['batch_size']
URBAN_META_PATH = dataset_config['meta_path']
URBAN_AUDIO_PATH = dataset_config['audio_path']
SR = spectrum_config['sr']  # 采样率


class UrbanSound8K(Dataset):
    def __init__(self, meta_csv_path, fold_list):
        self.X = []
        self.Y = []
        df = pd.read_csv(meta_csv_path)
        loop = tqdm(df['slice_file_name'])
        for filename in loop:
            fold_num = df[df['slice_file_name'] == filename]['fold'].values[0]
            if fold_num not in fold_list:
                continue
            fold = 'fold' + str(fold_num) + '/'
            full_path = os.path.join(URBAN_AUDIO_PATH, fold, filename)

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


# 制作数据集，返回波形，指定fold
def get_urbansound8k(fold_shuffle):
    if fold_shuffle:
        test_fold = random.randint(1, 10)
        train_fold = [i for i in range(1, 10) if i != test_fold]
        test_fold = [test_fold]
    else:
        train_fold = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_fold = [10]
    print('train fold : ', train_fold, 'test fold :', test_fold)
    UrbanSound8K_train = UrbanSound8K(URBAN_META_PATH, fold_list=train_fold)
    UrbanSound8K_test = UrbanSound8K(URBAN_META_PATH, fold_list=test_fold)
    train = DataLoader(UrbanSound8K_train, batch_size=BATCH_SIZE, shuffle=True)
    test = DataLoader(UrbanSound8K_test, batch_size=BATCH_SIZE, shuffle=False)
    return train, test


if __name__ == '__main__':
    UrbanSound8K_test = UrbanSound8K(URBAN_META_PATH, fold_list=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    test = DataLoader(UrbanSound8K_test, batch_size=BATCH_SIZE, shuffle=False)
    for x, y in test:
        print(x.shape)
