# -*- coding: utf-8 -*-
"""
@author: huangxs
@License: (C)Copyright 2021, huangxs
@CreateTime: 2021/11/16 19:10:00
@Filename: train

"""
import os
import numpy as np

# 设置临时环境变量，只输出error日志
from src.utils.metrics_util import accuracy_pixel_level

os.environ['GLOG_v'] = "3"
os.environ['DEVICE_ID'] = "3"

from collections import OrderedDict

import mindspore.dataset as ds

from src.cdnet import CDNet
from src.utils.dataset import MoNuSegGenerator, MoNuSegPreparedGenerator
from src.utils.direction_transform import get_transforms_list
from src.utils.loss import *

import glob
import numpy as np
import time

import mindspore
import mindspore.nn as nn
import mindspore.ops.functional as F
import mindspore.ops.operations as P
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common.initializer import One, Normal

from mindspore import context

# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
print('设置运行模式GPU')


class CDNetWithLoss(nn.Cell):
    def __init__(self, cdnet):
        super(CDNetWithLoss, self).__init__(auto_prefix=False)
        self._cdnet = cdnet

    def construct(self, _data, epoch, i):
        _input = _data['input']
        _target0 = _data['target0']
        _weight_map = _data['weight_map']
        _target_point = _data['target_point0'].squeeze(1)
        _target_direction0 = _data['target_direction0'].squeeze(1)

        _output_all = self._cdnet(_input)

        _output = _output_all[0]
        _output_point = _output_all[1]
        _output_direction = _output_all[2]
        _output_direction_0 = _output_direction

        # target | target0
        target = _target0
        if target.max() == 255:
            target = target // int(255 / 2)
        if target.dim() == 4:
            target = target.squeeze(1)

        target1 = _target0.asnumpy()
        num_classes = 3
        target_temp = np.zeros((_target0.shape[0], num_classes, _target0.shape[-2], _target0.shape[-1]), dtype=np.uint8)
        color_number = np.unique(target1)

        for j in range(_target0.shape[0]):
            target_temp[j, 0, :, :][target1[j, 0, :, :] == color_number[0]] = 1
            try:
                target_temp[j, 1, :, :][target1[j, 0, :, :] == color_number[1]] = 1
                if (num_classes == 3):
                    target_temp[j, 2, :, :][target1[j, 0, :, :] == color_number[2]] = 1
                else:
                    target_temp[j, 0, :, :][target1[j, 0, :, :] != color_number[0]] = 1
            except:
                if (num_classes != 1):
                    print('train IndexError: index 1 is out of bounds for axis 0 with size 1')

        _target0 = Tensor(target_temp, dtype=mstype.float32)
        #
        # # weight map
        _weight_map = _weight_map / 20
        if _weight_map.dim() == 4:
            _weight_map = _weight_map.squeeze(1)
        weight_map_var = _weight_map
        #
        # # target_direction | _target_direction0
        target_direction = _target_direction0
        direction_classes = 9
        target_direction_temp = np.zeros((_target_direction0.shape[0], direction_classes,
                                          _target_direction0.shape[-2], _target_direction0.shape[-1]))

        target_direction0_np = _target_direction0.astype("int32").asnumpy()
        target_np = target.astype("int32").asnumpy()

        for j in range(_target_direction0.shape[0]):
            unique_list = np.unique(target_direction0_np[j])
            unique_number = len(unique_list)

            if (unique_number > 1):  # Prevents images from appearing without instances
                for k in unique_list:
                    target_direction_temp[j, k, :, :][target_direction0_np[j, :, :] == k] = 1
                    target_direction_temp[j, k, :, :][((target_np[0, :, :] == 1) + (target_np[0, :, :] == 2)) == 0] = 0
            else:
                target_direction_temp[j, 0, :, :][target_direction0_np[j, :, :] == unique_list[0]] = 1

        _target_direction0 = Tensor(target_direction_temp, mstype.float32)

        ## compute loss
        _loss_ce = loss_ce(_output, target, weight_map_var)
        _loss_dice = loss_dice(_output, _target0)
        _loss_direction_CE, log_prob_maps_direction = loss_direction_ce(_output_direction_0, target_direction,
                                                                        weight_map_var)
        _loss_direction_dice = loss_direction_dice(_output_direction_0, _target_direction0)
        _loss_mse = loss_mse(_output_point, _target_point)

        loss = _loss_ce + _loss_dice + _loss_direction_CE + _loss_direction_dice + _loss_mse

        ## metric
        pred = np.argmax(log_prob_maps_direction.asnumpy(), axis=1)
        metrics = accuracy_pixel_level(pred, target_direction.asnumpy())

        # print(metrics)
        pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1, _ = metrics

        if i % 5 == 0:
            print(
                '''epoch:%3d, iter:%3d, loss=%.4f, l_ce=%.4f, l_dice=%.4f, l_d_CE=%.4f, l_d_dice=%.4f, l_mse=%.4f, pixel_accu=%.4f, p_iou=%.4f, p_recall=%.4f, p_precision=%.4f, p_F1=%.4f'''
                % (epoch, i, float(loss.asnumpy()), float(_loss_ce.asnumpy()), float(_loss_dice.asnumpy()),
                   float(_loss_direction_CE.asnumpy()), float(_loss_direction_dice.asnumpy()),
                   float(_loss_mse.asnumpy()), pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1))

        return loss


def run_train():
    print('start cdnet train')

    # ====== dataset ======
    train_data = MoNuSegPreparedGenerator(data_dir='data_prepare/train')
    dataset = ds.GeneratorDataset(train_data, ['input', 'weight_map', 'target0', 'target_point0', 'target_direction0'])
    dataset = dataset.batch(8)
    dataset = dataset.repeat(1)

    # ====== model ======
    print('modeling...')
    _cdnet = CDNet(backbone_name='vgg16_bn', encoder_freeze=False, classes=3)

    # if train from checkpoint file
    checkpoint_path = 'checkpoint/cdnet_vgg_pretrain_init.ckpt'
    if len(checkpoint_path) > 0:
        print('load checkpoint:', checkpoint_path)
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(_cdnet, param_dict)

    _cdnet_with_loss = CDNetWithLoss(_cdnet)

    # optimizer 用 adam
    optim = nn.Adam(_cdnet.trainable_params(), learning_rate=1e-3, beta1=0.9, beta2=0.999, weight_decay=1e-3)

    train_net = nn.TrainOneStepCell(_cdnet_with_loss, optim)
    train_net.set_train()

    loss = 0
    epochs = 301

    for epoch in range(epochs):
        for i, data in enumerate(dataset.create_dict_iterator()):
            if i == 0:
                time.sleep(5)
            loss = train_net(data, epoch, i)
        if epoch % 10 == 0:
            checkpoint_name = 'checkpoint/train_save/cdnet_epoch_%d_loss_%.2f.ckpt' % (epoch, float(loss.asnumpy()))
            save_checkpoint(_cdnet, checkpoint_name)

    return loss


if __name__ == "__main__":
    run_train()
