import torch
import dataset.datagenerator
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import random
import numpy as np
from model_src.cp_resnet_freq_damp import get_model_based_on_rho
from model_src.passt import passt
from model_src.mobilevit import mobileast_light, mobileast_light2, mobileast_cpresnet2
from model_src.cp_resnet import cp_resnet
from configs.dataconfig import dataset_config, spectrum_config
from configs.trainingconfig import normal_training_config
from configs.mixup import mixup_config
from configs.mixstyle import mixstyle_config
from dataset.datagenerator import TAU2022
from torch.utils.data import DataLoader
from train.normal_train import validate
import os
import torch
from tqdm import tqdm
from dataset.datagenerator import TAU2022_CLASSES, TAU2022_CLASSES_INVERT
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 使用最终评估集进行前向推理，并生成csv文件
def evaluate(model, eva_loader, savepath, device='cpu'):
    model.eval()
    csv_dict = {'filename': [],
                'scene_label': [],
                'airport': [],
                'bus': [],
                'metro': [],
                'metro_station': [],
                'park': [],
                'public_square': [],
                'shopping_mall': [],
                'street_pedestrian': [],
                'street_traffic': [],
                'tram': []}

    with torch.no_grad():
        loop = tqdm(eva_loader, desc='Validation')
        j = 0
        for x, filename in loop:
            if j > 50:
                break
            x = x.to(device)
            y_pred = F.softmax(model(x))  # 要求输出是每个类别的概率，使用softmax
            # print(y_pred.shape)   # (1, 10)
            y_ = torch.argmax(y_pred, dim=1)
            filename_ = str(filename.item()) + '.wav'  # h5无法存储字符串，存储的filename是文件编号
            csv_dict['filename'].append(filename_)
            for i in range(10):
                csv_dict[TAU2022_CLASSES_INVERT[i]].append(y_pred[0][i].item())
            csv_dict['scene_label'].append(TAU2022_CLASSES_INVERT[y_.item()])
            loop.set_description('Evaluation')
            j += 1

    df = pd.DataFrame(csv_dict)
    df.to_csv(savepath, index=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # 固定种子
    setup_seed(200)
    from dataset.datagenerator import get_valset, get_calibration_set, get_evaset
    from compress.quantization import quantization
    import model_src.quant_mobilevit
    from dataset.spectrum import ExtractMel

    extmel = ExtractMel(**spectrum_config)
    model = nn.Sequential(extmel, model_src.quant_mobilevit.mobileast_light2(mixstyle_config)).to('cpu')
    model.eval()
    weight_path = '../model_weights/best/passt+mobileastv3_light2 nomvit2 SiLU fc=160 DKD T=2 alpha=1 beta=8 MixStyle(0.3 0.6) Mixup(0.3 1) 1e-3(30) 1e-2(400)_0.55266.pt'
    model.load_state_dict(torch.load(weight_path))
    cal_dataloader = get_calibration_set(length=6400)
    quant_model = quantization(model, 'fx', cal_dataloader, show=True, backend='fbgemm')
    test_dataloader_list = get_valset()
    for scene_class, dataset in test_dataloader_list.items():
        val_loss, val_acc, test_device_info = validate(quant_model, dataset,
                                                       criterion=normal_training_config['criterion'], device='cpu')
        print('{}: val_loss={}, val_acc={}'.format(scene_class, val_loss, val_acc))
    eva_dataloader = get_evaset()
    evaluate(quant_model, eva_dataloader,
             savepath='../output/passt+mobileastv3_light2 nomvit2 SiLU fc=160 DKD T=2 alpha=1 beta=8 MixStyle(0.3 0.6) Mixup(0.3 1) 1e-3(30) 1e-2(400).csv')
