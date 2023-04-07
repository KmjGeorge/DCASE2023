import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import shutil
from tqdm import tqdm


def get_imagenetsub(index):
    path_train = 'D:/Datasets/ImageNet1k_sub{}/train/'.format(index)
    path_val = 'D:/Datasets/ImageNet1k/val/'
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=path_train,
        transform=data_transform)

    train_dataset_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    val_dataset = torchvision.datasets.ImageFolder(
        root=path_val,
        transform=data_transform)

    val_dataset_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

    return train_dataset_loader, val_dataset_loader


# 制作子集，每个类别只有100张图片
def make_imagenet1k_subset(index):
    train_path = 'D:/Datasets/ImageNet1k/train/'
    labels = os.listdir(train_path)
    os.mkdir(train_path.replace('ImageNet1k', 'ImageNet1k_sub{}'.format(index)).rstrip('train/'))
    os.mkdir(train_path.replace('ImageNet1k', 'ImageNet1k_sub{}'.format(index)))
    loop = tqdm(labels)
    for label in loop:
        loop.set_description('正在处理 %s' % label)
        for root, _, files in os.walk(os.path.join(train_path, label)):
            i = (index - 1) * 100
            dist_folder = root.replace('ImageNet1k', 'ImageNet1k_sub{}'.format(index)) + '/'
            os.mkdir(dist_folder)
            for filename in files:
                if i < index * 100:
                    pic_path = os.path.join(root, filename)
                    shutil.copy(pic_path, dist_folder)
                else:
                    continue
                i += 1


if __name__ == '__main__':
    make_imagenet1k_subset(1)
