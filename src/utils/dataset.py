# -*- coding: utf-8 -*-
"""
@author: huangxs
@License: (C)Copyright 2021, huangxs
@CreateTime: 2021/11/20 17:09:05
@Filename: dataset
service api views
"""
import mindspore
import numpy as np
import glob
import os
from PIL import Image

import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import Tensor


class MoNuSegGenerator:
    def __init__(self, image_dir, weight_dir, label_dir, transform_list):
        super(MoNuSegGenerator, self).__init__()
        self.transform_list = transform_list
        self.image_list = []
        self.weight_list = []
        self.label_list = []

        np.random.seed(58)
        _image_path_list = glob.glob(os.path.join(image_dir, '*.png'))
        for _image_path in _image_path_list:
            base_name = os.path.basename(_image_path).replace('.png', '')
            _weight_image = os.path.join(weight_dir, '%s_weight.png' % (base_name))
            _label_image = os.path.join(label_dir, '%s_label.png' % (base_name))
            if os.path.exists(_weight_image) and os.path.exists(_label_image):
                self.image_list.append(_image_path)
                self.weight_list.append(_weight_image)
                self.label_list.append(_label_image)

    def __getitem__(self, index):
        _image = Image.open(self.image_list[index]).convert('RGB')
        _weight = Image.open(self.weight_list[index])
        _label = Image.open(self.label_list[index]).convert('RGB')

        _sample_list = [_image, _weight, _label]

        # 执行transform
        for transform in self.transform_list:
            _sample_list = transform(_sample_list)

        # 合并数据，py_vision.ToTensor()将图片由HWC转置为CHW，再转为float后每个像素除以255，完成通道转换和归一化操作
        # 除第一个外，其余的都不归一化 @ 何红亮自定义了ToTensor
        _input = []
        for i, _sample in enumerate(_sample_list):
            if i == 0:
                _input.append(py_vision.ToTensor()(_sample))
            else:
                _input.append(py_vision.ToTensor()(_sample) * 255)

        # _input[3] = _input[3].squeeze(1)
        # _input[4] = _input[4].squeeze(1)

        # return _input
        return _input

    def __len__(self):
        return len(self.image_list)


class MoNuSegPreparedGenerator:
    def __init__(self, data_dir):
        super(MoNuSegPreparedGenerator, self).__init__()
        self.data_list = []

        np.random.seed(58)
        _data_path_list = glob.glob(os.path.join(data_dir, '*.npz'))
        for _data_path in _data_path_list:
            self.data_list.append(_data_path)

    def __getitem__(self, index):
        # load test data
        _data = []

        np_data = np.load(self.data_list[index])
        _data.append(Tensor(np_data['input'], mindspore.dtype.float32)[0])
        _data.append(Tensor(np_data['weight_map'], mindspore.dtype.float32)[0])
        _data.append(Tensor(np_data['target0'], mindspore.dtype.float32)[0])
        _data.append(Tensor(np_data['target_point0'], mindspore.dtype.float32)[0])
        _data.append(Tensor(np_data['target_direction0'], mindspore.dtype.float32)[0])

        # return _input
        return _data

    def __len__(self):
        return len(self.data_list)
