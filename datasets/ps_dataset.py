import os
import numpy as np
import torch
import torch.utils.data as data
from typing import Union
import cv2

from .builder import DATASETS
from .utils import _is_pan_image, get_image_id, load_image, data_normalize


# use the registry to manage the module
@DATASETS.register_module()
class PSDataset(data.Dataset):
    def __init__(self, image_dirs, bit_depth, norm_input=False):
        r""" Build dataset from folders

        Args:
            image_dirs (list[str]): image directories
            bit_depth (int): data value range in n-bit
            norm_input (bool): normalize the input to [0, 1]
        """
        super(PSDataset, self).__init__()

        self.image_dirs = image_dirs    #设置训练文件夹的位置（'data/PSData3/Dataset/GF-2/train_low_res'）
        self.bit_depth = bit_depth      #设置数据值的进制（10进制）
        self.norm_input = norm_input    #设置是否要进行归一化（FALSE）
        self.image_ids = []             #图像id
        self.image_prefix_names = []    #full-path filename prefix全路径文件名前缀
        for y in image_dirs:            #对训练文件夹下的图像进行遍历
            for x in os.listdir(y):     #os.listdir：获取文件夹下的图片
                if _is_pan_image(x):    #函数作用（有定义）判定是否为pan图像
                    self.image_ids.append(get_image_id(x))  #如果是pan图像 就获取图像的id并保存在image_ids下
                    self.image_prefix_names.append(os.path.join(y, get_image_id(x)))    #并把带图像id的路径保存在image_prefix_names下

    def __getitem__(self, index):
        # type: (int) -> dict[str, Union[torch.Tensor, str]]
        prefix_name = self.image_prefix_names[index]    #'data/PSData3/Dataset/GF-2/train_low_res/4'

        input_dict = dict(
            input_lr=load_image('{}_lr.tif'.format(prefix_name)),  # [4,64,64] LR MS
            input_pan=load_image('{}_pan.tif'.format(prefix_name))[np.newaxis, :],  # [1,256,256] PAN
        )
        if os.path.exists('{}_mul.tif'.format(prefix_name)) and len(self.image_dirs) == 1:
            input_dict['target'] = load_image('{}_mul.tif'.format(prefix_name))  # [4,256,256] HR MS gt

        # [1,64,64] Gaussian Degraded PAN 高斯下采样的PAN
        input_dict['input_pan_l'] = cv2.pyrDown(cv2.pyrDown(input_dict['input_pan'][0]))[np.newaxis, :]

        for key in input_dict:  # numpy2torch
            input_dict[key] = torch.from_numpy(input_dict[key]).float()

        if self.norm_input:
            input_dict = data_normalize(input_dict, self.bit_depth)

        input_dict['image_id'] = self.image_ids[index]
        return input_dict

    def __len__(self):
        return len(self.image_ids)
