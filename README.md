# DCASE2023 Task1

施工中……

requirements.txt为项目主要使用的包，可能有遗漏，缺哪个再装哪个

pytorch版本请低于2.0

此外需安装hear21passt，原项目地址https://github.com/kkoutini/passt_hear21

或用以下命令安装

```
pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt' 
```

# 此Readme为旧版，等待更新

## I. 文件结构

```
PROJECT ROOT
│   README.md	
|   requirements.txt
└───compress	        ========>  存放模型压缩策略(如量化)		
│   │   quantization.py(未完成)

└───configs	        ========>  存放配置文件
│   │   dataconfig.py	 			数据集及频谱参数配置
│   │   trainingconfig.py   			训练参数配置
│   │   mixup.py   				mixup参数
│   │   mixstyle.py   				mixstyle参数

└───dataset             ========>  存放数据处理相关					
│   │   augmentation.py					数据增强策略	
│   │   datagenerator.py				数据集制作(以h5存储)和读取（返回Datagenerator供网络训练）
│   │   spectrum.py					频谱特征提取，是一个继承nn.Module的类，作为网络的输入层使用
│   │   files_reassembled.ipynb			        制作reassembled数据集（10s），拼接音频文件
|   |   meta_csv_reassembled.ipynb		        制作reassembled数据集（10s），拼接meta文件
│   └───h5      ========>  存放数据集的h5存储（文件过大不好上传，请运行datagenerator.py制作，完成后应当有以下文件）
|   |	|   tau2022_train.h5			
|   |   |   tau2022_test.h5
|   |   |   tau2022_reassembled_train.h5
|   |   |   tau2022_reassembled_test.h5

└───figure		========>  存放训练图表		
│   │   (empty)		

└───logs		========>  存放训练日志
│   │   (empty)

└───model_src	        ========>  存放模型定义
|   │   acdnet.py					ACDNet(https://doi.org/10.1016/j.patcog.2022.109025)
|   │	cp_resnet.py					CP_ResNet(https://arxiv.org/abs/2105.12395v1)，此为去年第一使用的配置
|   |	cp_resnet_freq_damp.py	                        CP_ResNet 使用频率上的damp，目前作为教师使用
|   |	mobilevit.py					MobileViT(https://arxiv.org/abs/2110.02178v2)，并把它迁移到音频上
|   |	passt.py					PaSST(https://arxiv.org/abs/2110.05069)，此处使用hear21passt库的实现
|   				 
|   └───module		========> 网络结构使用到的模块请放到此处
|	|	| 	mixstyle.py				MixStyle，添加参数freq=True可使用频率上的MixStyle
|	|	| 	rfn.py					松弛频率归一化 Relaxed instance frequency-wise normalization
|       |       |	ssn.py					子谱归一化 Sub-spectrum Normalization
|       |       |       grl.py                                  梯度反转层，用于域对抗训练

└───model_weights       ========> 存放模型参数
|	└───pretrained	========> 存放预训练参数

└───optim		========> 自定义优化器和学习率调度器
|	|	scheduler.py	自定义学习率调度器
|	|	test.py		测试调度器，绘制学习率曲线

└───size_cal	        ========> 官方给出的模型复杂度计算器
|	...
|	...

└───train		========> 定义训练过程
|	|	normal_train.py					定义普通训练过程
|	|	distillation.py					定义知识蒸馏过程
|       |       domain_adversarial.py                           定义域对抗训练过程(未完成)

└───run                 ========> 定义运行流程
|	|	run.py						训练某一模型，读取配置configs.trainingconfig.normal_training_config
|       |       dis_run.py                                      蒸馏某一模型，读取配置configs.trainingconfig.distillation_config
|       |       evaluate.py                                     评估，使用验证集计算准确率
|	|	testscript.py					测试脚本，开发用
```



## II. 数据准备

#### 1、使用dataset下的两个ipynb，获得reassembled数据集，请将路径修改为本地存放路径

#### 2、在configs/dataconfig.py中修改数据集路径

​		meta_path  audio_path分别数据集meta.csv路径、meta.csv所在的同级目录(不进到audio文件夹)

​		h5path为制作的h5个格式数据路径

#### 3、制作h5格式的数据集

##### ①设置数据采样率

​		configs.dataconfig.spectrum_config ['sr']，默认为32000

##### ②制作原版数据集

​		<1>将configs/dataconfig的data_config更改为：

```python
dataset_config = {
    'name': 'tau2022',
    'batch_size': 64,
    'meta_path': 'D:/datasets/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv',  # 你的数据集meta.csv路径
    'audio_path': 'D:/datasets/TAU-urban-acoustic-scenes-2022-mobile-development/',  # 你的数据集路径 meta.csv所在目录
    'h5path': 'D:/github/DCASE2023/dataset/h5/',  # 想要保存的h5格式的数据集位置
    'num_classes': 10,
    'shuffle': True,
}
```

​		<2>运行datagenerator.py

##### ③reassembled版数据集

​		<1>将configs/dataconfig的data_config更改为：

```python
dataset_config = {
    'name': 'tau2022_reassembled',
    'batch_size': 64,
    'meta_path': 'D:/datasets/TAU-urban-acoustic-scenes-2022-mobile-development-reassembled/meta.csv', # 你的数据集meta.csv路径
    'audio_path': 'D:/datasets/TAU-urban-acoustic-scenes-2022-mobile-development-reassembled/',  #你的数据集meta.csv所在目录
    'h5path': 'D:/github/DCASE2023/dataset/h5/',  # 想要保存的h5格式的数据集位置
    'num_classes': 10,
    'shuffle': True,
}
```

​		<2>运行datagenerator.py

#####  ④完成后，h5path下应当有如下文件：

```
└───data_config['h5path']
|   |  tau2022_train.h5			
|   |  tau2022_test.h5
|   |  tau2022_reassembled_train.h5
|   |  tau2022_reassembled_test.h5
```



#### 4、自行训练如何获取数据集（Dataloader对象）

##### ①获取原版TAU2022

```python
# 使用前请把dataset/dataconfig.py/dataset_config中改为"tau2022"版本
from dataset.datagenerator import get_tau2022
TAU2022_train, TAU2022_test = get_tau2022()

i = 0
# 输出5个数据的shape
for x, y in TAU2022_train:
    if i == 5:
        break
    print(x.shape)
    print(y.shape)
    i += 1
```

##### ②获取10s版TAU2022

```python
# 使用前请把dataset/dataconfig.py/dataset_config中改为"tau2022_reassembled"版本
from dataset.datagenerator import get_tau2022_reassembled
TAU2022_reassembled_train, TAU2022_reassembled_test = get_tau2022_reassembled()
...
```

##### ③获取10s版TAU2022，但每次送入网络的是一个随机1s切片

```python
# 使用前请把dataset/dataconfig.py/dataset_config中改为"tau2022_random_slicing"版本
from dataset.datagenerator import get_tau2022_reassembled_random_slicing
Random_Slicing_train, Random_Slicing_test = get_tau2022_reassembled_random_slicing()
...
```



## III. 模型获取

#### 1、如何获取模型

```python
from configs.mixstyle import mixstyle_config  # 每个模型可选择是否使用mixstyle（仅在输入时，若想在其他地方使用，请在网络结构中添加MixStyle层）
from model_src.acdnet import GetACDNetModel
from model_src.cp_resnet import cp_resnet
from model_src.mobilevit import mobileast_light
from model_src.passt import passt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 具体模型结构请前往model_src下的实现查看，尚未建立模型配置文件
acdnet = GetACDNetModel(mixstyle_conf=mixstyle_config, input_len=32000, nclass=10, sr=32000).to(device)
cp_resnet = cp_resnet(mixstyle_conf=mixstyle_config).to(device)
mobileast_light = mobileast_light(mixstyle_conf=mixstyle_config).to(device)
passt = passt(mixstyle_conf=mixstyle_config, n_classes=10).to(device)
```

#### 2、如何计算复杂度

```python
# 计算模型大小，使用DCASE官方工具
from size_cal import nessi
nessi.get_model_size(model, 'torch', input_size=(1, channel, frequency, time)) 
```

#### 3、获取频谱提取层

```python
from dataset.spectrum import ExtractMel
from configs.dataconfig import spectrum_config
# 若要修改频谱提取策略，请修改该类的实现
extmel = ExtractMel(**spectrum_config)
```

#### 4、已训练的模型参数

##### PaSST audioset预训练参数(passt_s_swa_p16_128_ap476) + TAU2022上训练的参数(val_acc=59.87)

链接：https://pan.baidu.com/s/1QNbvlRYhAk6fuUP48RFnJA 
提取码：2yl2

##### TAU2022上的训练配置：

​	**数据**：tau2022_random_slicing

​	**频谱提取**（见model_src.passt)：

```python
AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                               timem=20,
                               htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1,
                               fmax_aug_range=1)
```

​	**模型配置**：

```python
get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=10, in_channels=1, fstride=10,
                            tstride=10,
                            input_fdim=128, input_tdim=100, u_patchout=0, s_patchout_t=0, s_patchout_f=6)
```

​    由于输入尺寸变化，使用自适应平均池化整合预训练权重至对应的shape

​	**其他**：mixup（alpha=0.3)，mixstyle(alpha=0.3，p=0.6)，AdamW(weight_decay=1e-3)，共250epochs，前30个epoch学习率warmup至1e-5，而后余弦退火至1e-7



## IV. 继续添加模块

​		新增数据集请在/**dataset/datagenerator.py**中加入新的类，并修改配置文件/**configs/dataconfig.py**

​		新增数据增强方案请添加到**augmentation.py**

​		新增频谱提取策略和增强方案请添加到**spectrum.py**（为了方便，封装为网络输入层，参考ExtractMel类，继承nn.Module类）

​		新增模型实现请添加到/**model_src**，使用的模块可以放到/**model_src/module**

​		如果有预训练权重放到**model_weights/pretrained**

​		新增优化器或lr调度器请添加到/**optim**

​		新增自定义训练行为（如对抗训练、知识蒸馏）请添加到/**train**



## V. 运行示例

##### ①修改配置

​		configs/mixup.py 指定是否使用mixup，超参数alpha

​		configs/mixstyle.py 指定是否使用mixstyle，超参数alpha, p

​		configs/dataconfig.py 指定数据集

​		configs/trainingconfig，指定epoch、模型、优化器和lr调度器

##### ②运行程序**run/run.py**

​		示例程序是训练单个模型，使用知识蒸馏请修改相关代码（调用train/下的不同函数)

​		运行过程产生参数文件存放于**model_weights**/，日志文件存放于**logs**/，训练结束后将图像至**figure**/

​	



​	









