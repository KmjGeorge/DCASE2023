from hear21passt.base import get_basic_model, get_model_passt, load_model
import torch


def passt():
    model = get_basic_model(mode="logits")  # model包含model.mel和model.net 分别是输入特征提取层和主干网络
    passt = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=10)
    # 加载预训练权重，网络结构有变化的剔除
    pretrained_dict = torch.load('../model_weights/pretrained/passt-s-f128-p16-s10-ap.476-swa.pt')
    passt_dict = passt.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in passt_dict}
    passt_dict.update(pretrained_dict)

    # 修改最后一层权重适应新的分类数量
    head1_weight = pretrained_dict['head.1.weight']
    head_dist_weight = pretrained_dict['head_dist.weight']
    head1_bias = pretrained_dict['head.1.bias']
    head_dist_bias = pretrained_dict['head_dist.bias']
    passt.load_state_dict(passt_dict)

    model.net = passt
    model.train()
    model.cuda()
    return model


if __name__ == '__main__':
    from size_cal import nessi

    model = passt()

    nessi.get_model_size(model, 'torch', input_size=(1, 32000))
