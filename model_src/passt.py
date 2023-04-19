from hear21passt.base import get_basic_model, get_model_passt, load_model
from hear21passt.models.preprocess import AugmentMelSTFT
import hear21passt.models.passt
import torch
import torch.nn.functional as F
import torch.nn as nn
from model_src.module.mixstyle import MixStyle


# 加载预训练权重，网络结构有变化的，对参数作调整
def load_pretrained_weights(passt, n_classes):
    pretrained_dict = torch.load('../model_weights/pretrained/passt-s-f128-p16-s10-ap.476-swa.pt')
    passt_dict = passt.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in passt_dict}

    ''' 寻找shape被更改的层
    for (k1, v1), (k2, v2) in zip(pretrained_dict.items(), passt_dict.items()):
        if v1.shape != v2.shape:
            print(k1, v1.shape, v2.shape)
    assert False
    '''
    passt_dict.update(pretrained_dict)
    # 利用自适应平均池化整合位置编码
    time_new_pos_embed = passt_dict['time_new_pos_embed']  # (1, 768, 1, 99)
    time_new_pos_embed = torch.unsqueeze(F.adaptive_avg_pool1d(time_new_pos_embed[0], output_size=10),
                                         dim=0)  # (1, 768, 1, 10)

    # 修改最后的分类层权重适应新的分类数量  527类->10类
    head1_weight = passt_dict['head.1.weight']  # (527, 768)
    head_dist_weight = passt_dict['head_dist.weight']
    head1_bias = passt_dict['head.1.bias']  # (527,)
    head_dist_bias = passt_dict['head_dist.bias']

    # adaptive_avg_pool1d修改的是最后一个维度，需要转置 (527, 768) -> (768, 527) -> (768, 10) -> (10, 768)
    head1_weight = F.adaptive_avg_pool1d(head1_weight.T, output_size=n_classes).T

    head_dist_weight = F.adaptive_avg_pool1d(head_dist_weight.T, output_size=n_classes).T

    # (527,) -> (1, 527) -> (1, 10) -> (10,)
    head1_bias = torch.unsqueeze(head1_bias, 0)
    head1_bias = F.adaptive_avg_pool1d(head1_bias, output_size=n_classes)
    head1_bias = torch.squeeze(head1_bias)

    head_dist_bias = torch.unsqueeze(head_dist_bias, 0)
    head_dist_bias = F.adaptive_avg_pool1d(head_dist_bias, output_size=n_classes)
    head_dist_bias = torch.squeeze(head_dist_bias)

    passt_dict['time_new_pos_embed'] = time_new_pos_embed
    passt_dict['head.1.weight'] = head1_weight
    passt_dict['head_dist.weight'] = head_dist_weight
    passt_dict['head.1.bias'] = head1_bias
    passt_dict['head_dist.bias'] = head_dist_bias

    passt.load_state_dict(passt_dict)


def passt(mixstyle_conf, pretrained=True, n_classes=10):
    model = get_basic_model(mode="logits")  # model包含model.mel和model.net 分别是输入特征提取层和主干网络
    ''' # model.mel默认为
    AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                              timem=192,
                              htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                              fmax_aug_range=2000)
    '''
    model.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                               timem=20,
                               htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1,
                               fmax_aug_range=1)
    """ 
    :param arch: Base ViT or Deit architecture
    :param pretrained: use pretrained model on imagenet
    :param n_classes: number of classes
    :param in_channels: number of input channels: 1 for mono
    :param fstride: the patches stride over frequency.
    :param tstride: the patches stride over time.
    :param input_fdim: the expected input frequency bins.
    :param input_tdim: the expected input time bins.
    :param u_patchout: number of input patches to drop in Unstructured Patchout as defined in https://arxiv.org/abs/2110.05069
    :param s_patchout_t: number of input time frames to drop Structured Patchout as defined in https://arxiv.org/abs/2110.05069
    :param s_patchout_f:  number of input frequency bins to drop Structured Patchout as defined in https://arxiv.org/abs/2110.05069
    :param audioset_pretrain: use pretrained models on Audioset.
    """
    passt = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=n_classes, in_channels=1, fstride=10,
                            tstride=10,
                            input_fdim=128, input_tdim=100, u_patchout=0, s_patchout_t=0, s_patchout_f=6)
    if pretrained:
        load_pretrained_weights(passt, n_classes)

    if mixstyle_conf['enable']:
        model.net = nn.Sequential(MixStyle(mixstyle_conf['p'], mixstyle_conf['alpha'], mixstyle_conf['freq']), passt)
    else:
        model.net = passt
    model.train()
    return model


if __name__ == '__main__':
    from size_cal import nessi

    model = passt(10)
    param = model.net.state_dict()
    nessi.get_model_size(model, 'torch', input_size=(1, 32000))
    # print(param['1.time_new_pos_embed'].shape)
