# -*- coding: utf-8 -*-
"""
@author: huangxs
@License: (C)Copyright 2021, huangxs
@CreateTime: 2021/11/25 19:17:09
@Filename: metrics_util
service api views
"""

import numpy as np
import math
import random
import time


def accuracy_pixel_level(output, target):
    """ Computes the accuracy during training and validation for ternary label """
    batch_size = target.shape[0]
    results = np.zeros((6,), np.float)

    for i in range(batch_size):
        pred = output[i, :, :]
        label = target[i, :, :]

        # inside part
        pred_inside = pred == 1
        label_inside = label == 1
        metrics_inside = compute_pixel_level_metrics(pred_inside, label_inside)

        results += np.array(metrics_inside)

    return [value / batch_size for value in results]


def compute_pixel_level_metrics(pred, target):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """
    start_time = time.time()
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    tp = np.sum(pred * target)  # true postives
    tn = np.sum((1 - pred) * (1 - target))  # true negatives
    fp = np.sum(pred * (1 - target))  # false postives
    fn = np.sum((1 - pred) * target)  # false negatives

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    performance = (recall + tn / (tn + fp + 1e-10)) / 2
    iou = tp / (tp + fp + fn + 1e-10)

    return [acc, iou, recall, precision, F1, performance]


class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
