# DCASE2023 Task1

施工中……



## I. 文件结构

### /dataset

##### 	数据处理代码，包括：

​	datagenerator.py  —  数据集制作（建议存储为h5，直接读取音频时间过长，存放到dataset/h5/)

​	files_reassemble.ipynb,  meta_csv_reassemble.ipynb — 按去年第1方案，制作reassembled数据集(1s拼接为10s)

​	augmentation.py — 数据增强策略，目前只实现了mixup

​    spectrum.py — 提取频谱特征和频谱增强，封装为网络的输入层，目前代码搬的去年第1方案(mel滤波器随机偏移+specaugment)

​    dataconfig.py  —  数据集和频谱参数配置

### /compress 

##### 		存放压缩策略

### /model_src

##### 	存放模型结构，使用的模块可以定义在/model_src/module

​	目前定义的模型：

​	cp_resnet.py — 去年第一(CPJKU)使用的rfr-cnn(学生)

​	rfr-cnn.py — 去年第三(RECEPTIVE FIELD REGULARIZED CNNS WITH TRADITIONAL AUDIO AUGMENTATIONS)使用的rfr-cnn(教师)

​	mobilevit.py — mobilevit，包含s, xs, xxs三个版本，参数量依次递减

### /model_weights

  ##### 			存放训练参数

### /optim

  ##### 			自定义优化器和学习率调度器

​	scheduler.py — 目前使用了warmup，见GradualWarmupScheduler类

​	test.py — 测试样例

### /train

##### 		自定义训练行为

​	normal_train.py — 普通的训练和验证（可选择mixup)

​	distillation.py — 蒸馏

​	trainingconfig.py — 训练参数配置

### /size_cal

##### 官方提供的计算模型大小的工具，用法：

```python
from size_cal import nessi
nessi.get_model_size(model, model_type='torch', input_size)
# 如果频谱提取被集成到网络当中，则input_size = (batch, sr*time)   batch取1， sr为采样率， time为音频时长
```

### /logs

##### 	存放训练日志文件

### /figure

##### 存放训练曲线和其他图

### /run

##### 		主程序

​	run.py — 主程序示例，流程：读取配置—获取模型—计算模型大小—获取数据集—训练—保存日志



## II. 数据准备

#### 1、使用dataset下的两个ipynb制作reassembled数据集(1s拼接为10s)

#### 2、在dataset/dataconfig.py中修改数据集路径

​		meta_path、audio_path为原版数据集meta.csv路径、meta.csv所在的同级目录(不进到audio文件夹)

​		reassembled_meta_path、reassembled_audio_path为reassembled数据集对应路径

#### 3、制作h5格式的数据集，包括①原版  ②将reassembled版随机1s切片若干次得到的版本

​		h5path 定义原版h5存储路径（是个文件夹，包含训练集和测试集）

​		slicing_h5path定义切片版存储路径（精确到文件名，只需制作训练集）

​		运行datagenerator.py

#### 4、自行训练如何获取数据集

```python
# 调用前确保dataconfig.py/dataset_config中的参数正确
TAU2022_train, TAU2022_test = get_tau2022()  # 获得原版TAU2022数据集
Random_Slicing_train, Random_Slicing_test = get_tau2022_random_slicing()  	# 获得切片版TAU2022数据集
# 返回的类型均为Dataloader
```



## III. 运行主程序示例

#### 1、在train/trainingconfig.py中修改训练参数

​	目前只加了几个参数，而有关优化器、学习率等定义在train文件夹下，方便新增自定义训练过程

#### 2、运行run.py

​	别看结果了，先能run起来再说



## IV. 继续添加模块

​	新增数据集请在**/dataset/datagenerator.py**中加入新的类；新增数据增强方案请添加到**augmentation.py**；新增频谱提取策略和增强方案请添加到**spectrum.py**（为了方便，封装为网络输入层，参考ExtractMel类，继承nn.Module类）

​	新增模型请添加到**/model_src**，如果有预训练权重放到**model_weights/pretrained**

​	新增优化器或lr调度器请添加到**/optim**

​	新增自定义训练行为（如蒸馏）请添加到**/train**















