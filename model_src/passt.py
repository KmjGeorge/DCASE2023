from hear21passt.base import get_basic_model, get_model_passt, load_model
import torch
import torch.nn.functional as F


def passt(n_classes=10):
    model = get_basic_model(mode="logits")  # model包含model.mel和model.net 分别是输入特征提取层和主干网络
    passt = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=n_classes)
    # 加载预训练权重，网络结构有变化的剔除
    pretrained_dict = torch.load('../model_weights/pretrained/passt-s-f128-p16-s10-ap.476-swa.pt')
    passt_dict = passt.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in passt_dict}
    passt_dict.update(pretrained_dict)

    # 修改最后一层权重适应新的分类数量 利用自适应平均池化 527类->10类
    head1_weight = passt_dict['head.1.weight']  # (527, 768)
    head_dist_weight = passt_dict['head_dist.weight']
    head1_bias = passt_dict['head.1.bias']  # (527,)
    head_dist_bias = passt_dict['head_dist.bias']

    head1_weight = F.adaptive_avg_pool1d(head1_weight.T, output_size=n_classes).T  # adaptive_avg_pool1d修改的是最后一个维度，先转置再转置回来

    head_dist_weight = F.adaptive_avg_pool1d(head_dist_weight.T, output_size=n_classes).T

    head1_bias = torch.unsqueeze(head1_bias, 0)  # (1, 527)
    head1_bias = F.adaptive_avg_pool1d(head1_bias, output_size=n_classes)
    head1_bias = torch.squeeze(head1_bias)

    head_dist_bias = torch.unsqueeze(head_dist_bias, 0)
    head_dist_bias = F.adaptive_avg_pool1d(head_dist_bias, output_size=n_classes)
    head_dist_bias = torch.squeeze(head_dist_bias)

    passt_dict['head.1.weight'] = head1_weight
    passt_dict['head_dist.weight'] = head_dist_weight
    passt_dict['head.1.bias'] = head1_bias
    passt_dict['head_dist.bias'] = head_dist_bias

    passt.load_state_dict(passt_dict)

    model.net = passt
    model.train()
    model.cuda()
    return model


if __name__ == '__main__':
    from size_cal import nessi
    model = passt(10)
    param = model.net.state_dict()
    nessi.get_model_size(model, 'torch', input_size=(1, 32000))

