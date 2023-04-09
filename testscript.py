'''
随便测试点什么
'''
import os
import numpy as np
import torch
import torch.nn.functional as F

# path = 'G:/datasets/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv'
# print(os.path.split(path))


# import h5py
# h5_path = '123.h5'
#
#
# y = ['airport', 'bus', 'metro']
# for label in y:
#     with h5py.File(h5_path, 'a') as f:
#         f.create_group(label)
#     x_list = []
#     for i in range(10):
#         x_new = np.random.random(size=(3,))
#         x_list.append(x_new)
#     with h5py.File(h5_path, 'a') as f:
#         f[label].create_dataset('x', data=x_list)
#
#
# with h5py.File(h5_path, 'r') as f:
#     for k, v in f.items():
#         print(k, v)
#         for label, x in f[k].items():
#             print('    ', label, x)

# x = [i for i in range(10)]
# y = [i+100 for i in range(10)]
#
# for a, b in zip(x, y):
#     print(a, b)

# import pandas as pd
# df = pd.read_csv('G:\datasets\TAU-urban-acoustic-scenes-2022-mobile-development\evaluation_setup\\fold1_train.csv', sep='\t')
# small_df = df[0:800]
# small_df.to_csv('G:\datasets\TAU-urban-acoustic-scenes-2022-mobile-development\evaluation_setup\\fold1_train_small.csv', index_label=False)
meta_new = "D:/datasets/TAU-urban-acoustic-scenes-2022-mobile-development-reassembled/meta.csv"
print(os.path.split(meta_new)[0])
os.mkdir(os.path.join(os.path.split(meta_new)[0], 'evaluation_setup'))


