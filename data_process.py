# -*- coding: utf-8 -*-
"""
@author: huangxs
@License: (C)Copyright 2021, huangxs
@CreateTime: 2021/11/16 19:10:00
@Filename: data process

"""
import numpy as np
from collections import OrderedDict
from src.utils.dataset import MoNuSegGenerator, MoNuSegPreparedGenerator
from src.utils.direction_transform import get_transforms_list

import mindspore.dataset as ds


def data_processing():
    transform_dict = OrderedDict([('random_color', 1), ('horizontal_flip', True), ('vertical_flip', True),
                                  ('random_chooseAug', 1), ('random_crop', 256),
                                  ('label_encoding', [3, 2, 1])])
    transform_list = get_transforms_list(transform_dict)

    # 构建用于训练的dataset
    monuseg = MoNuSegGenerator(
        image_dir='data/MoNuSeg_oridata/images/train_300',
        weight_dir='data/MoNuSeg_oridata/weight_maps/train_300',
        label_dir='data/MoNuSeg_oridata/labels/train_300',
        transform_list=transform_list)
    dataset = ds.GeneratorDataset(monuseg, ['input', 'weight_map', 'target0', 'target_point0', 'target_direction0'])
    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)

    # 测试构建的数据集迭代
    for i, data in enumerate(dataset.create_dict_iterator()):
        print('i=%d' % i, data['input'].shape, data['weight_map'].shape,
              data['target0'].shape, data['target_point0'].shape, data['target_direction0'].shape)
        np.savez('data_prepare/train/train_data_%d' % i, input=data['input'].asnumpy(),
                 weight_map=data['weight_map'].asnumpy(), target0=data['target0'].asnumpy(),
                 target_point0=data['target_point0'].asnumpy(), target_direction0=data['target_direction0'].asnumpy())

    print('finish data processing before train')


if __name__ == "__main__":
    data_processing()
