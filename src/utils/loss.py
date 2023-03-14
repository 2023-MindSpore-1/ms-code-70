# -*- coding: utf-8 -*-
"""
@author: huangxs
@License: (C)Copyright 2021, huangxs
@CreateTime: 2021/11/24 10:59:41
@Filename: loss
service api views
"""

import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops.functional as F
import mindspore.ops.operations as P
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.common.initializer import One, Normal


def nll_loss_2d(_log_prob_maps, _target_var, nll_criterion):
    _shape = _log_prob_maps.shape
    log_prob_maps_transpose = _log_prob_maps.transpose(0, 2, 3, 1)
    nll_logits = log_prob_maps_transpose.view(-1, _shape[1])
    nll_label = _target_var.view(-1)

    nll_weight = Tensor(np.ones(_shape[1]), dtype=mstype.float32)
    nll_loss = nll_criterion(nll_logits, nll_label.astype('Int32'), nll_weight)
    _loss_map = nll_loss[0].view(_shape[0], _shape[2], _shape[3])

    return _loss_map


class DiceLoss(nn.Cell):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def construct(self, input, target):
        #         print(target.shape)
        N = target.shape[0]
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Cell):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def construct(self, input, target, weights=None):
        C = target.shape[1]
        # if weights is None:
        # weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


def loss_ce(_output, _target, _weight):
    # 1. loss_CE
    _shape = _output.shape
    _output = _output.transpose(0, 2, 3, 1).view(-1, _shape[1])
    log_prob_maps = nn.LogSoftmax(axis=1)(_output)
    log_prob_maps = log_prob_maps.view(_shape[0], _shape[2], _shape[3], _shape[1]).transpose(0, 3, 1, 2)

    criterion = ops.NLLLoss(reduction='none')
    loss_map = nll_loss_2d(log_prob_maps, _target, criterion)
    loss_map = loss_map * _weight
    _loss_CE = loss_map.mean()
    return _loss_CE


def loss_dice(_output, _target):
    # 2. loss_dice
    _shape = _output.shape
    _output = _output.transpose(0, 2, 3, 1).view(-1, _shape[1])
    prob_maps = nn.Softmax(axis=1)(_output)
    prob_maps = prob_maps.view(_shape[0], _shape[2], _shape[3], _shape[1]).transpose(0, 3, 1, 2)

    criterion_dice = MulticlassDiceLoss()
    _loss_dice = criterion_dice(prob_maps, _target)
    return _loss_dice


def loss_direction_ce(_output_direction_0, _target_direction, _weight_map_var):
    # 3. loss_direction_CE
    _shape = _output_direction_0.shape
    _output_direction_0 = _output_direction_0.transpose(0, 2, 3, 1).view(-1, _shape[1])
    log_prob_maps_direction = nn.LogSoftmax(axis=1)(_output_direction_0)
    log_prob_maps_direction = log_prob_maps_direction.view(_shape[0], _shape[2], _shape[3],
                                                           _shape[1]).transpose(0, 3, 1, 2)

    criterion = ops.NLLLoss(reduction='none')
    loss_direction_map = nll_loss_2d(log_prob_maps_direction, _target_direction, criterion)

    loss_direction_map *= _weight_map_var

    _loss_direction_CE = loss_direction_map.mean()
    return _loss_direction_CE, log_prob_maps_direction


def loss_direction_dice(_output_direction_0, _target_direction0):
    # 4. loss_direction_dice
    _shape = _output_direction_0.shape
    _output_direction_0 = _output_direction_0.transpose(0, 2, 3, 1).view(-1, _shape[1])
    prob_maps_direction = nn.Softmax(axis=1)(_output_direction_0)
    prob_maps_direction = prob_maps_direction.view(_shape[0], _shape[2], _shape[3],
                                                   _shape[1]).transpose(0, 3, 1, 2)

    criterion_dice = MulticlassDiceLoss()
    _loss_direction_dice = criterion_dice(prob_maps_direction, _target_direction0)

    return _loss_direction_dice


def loss_mse(_output_point, _target_point):
    # 5. loss_mse
    _target_point = _target_point.view(8, 1, 256, 256)
    _loss_mse = nn.MSELoss()(_output_point, _target_point)

    return _loss_mse
