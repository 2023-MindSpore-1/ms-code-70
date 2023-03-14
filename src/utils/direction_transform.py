# -*- coding: utf-8 -*-
"""
@author: huangxs
@License: (C)Copyright 2021, huangxs
@CreateTime: 2021/11/19 16:01:02
@Filename: direction_transform
service api views
"""
import mindspore.nn as nn
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import dtype as mstype
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
from mindspore.common.initializer import One, Normal

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random
import albumentations as albu
import numbers
import time
import copy
from skimage import morphology, io, color, measure, feature
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, binary_dilation, binary_fill_holes
import math
from collections import OrderedDict
import numpy as np

import mindspore.ops as ops
import os
from mindspore import dtype as mstype
from mindspore import Tensor

import mindspore.ops as ops


class RandomColor(object):
    def __init__(self, randomMin=1, randomMax=2):
        self.randomMin = randomMin
        self.randomMax = randomMax

    def __call__(self, imgs):
        out_imgs = list(imgs)
        img = imgs[0]
        # 0.5
        random_factor = 1 + (np.random.rand() - 0.5)

        color_image = ImageEnhance.Color(img).enhance(random_factor)
        random_factor = 1 + (np.random.rand() - 0.5)

        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = 1 + (np.random.rand() - 0.5)

        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = 1 + (np.random.rand() - 0.5)

        img_output = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

        out_imgs[0] = img_output

        return tuple(out_imgs)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """

        pics = []
        if random.random() < 0.5:
            for img in imgs:  # imgs
                pics.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            return tuple(pics)
        else:
            return imgs


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        pics = []
        if random.random() < 0.5:
            for img in imgs:
                pics.append(img.transpose(Image.FLIP_TOP_BOTTOM))
            return tuple(pics)
        else:
            return imgs


class RandomElasticDeform(object):
    """ Elastic deformation of the input PIL Image using random displacement vectors
        drawm from a gaussian distribution
    Args:
        sigma: the largest possible deviation of random parameters
    """

    def __init__(self, num_pts=4, sigma=20):
        self.num_pts = num_pts
        self.sigma = sigma

    def __call__(self, imgs):
        pics = []

        do_albu = 1
        if (do_albu == 1):
            image = np.array(imgs[0])
            weightmap = np.expand_dims(imgs[1], axis=2)
            label = np.array(imgs[2])  # np.expand_dims(imgs[2], axis=2)
            if (len(label.shape) == 2):
                label = label.reshape(label.shape[0], label.shape[1], 1)
            concat_map = np.concatenate((image, weightmap, label), axis=2)

            transf = albu.ElasticTransform(always_apply=False, p=1.0, alpha=1.0, sigma=50, alpha_affine=50,
                                           interpolation=0, border_mode=0,
                                           value=(0, 0, 0),
                                           mask_value=None, approximate=False)  # border_mode 用于指定插值算法

            concat_map_transf = transf(image=concat_map)['image']
            image_transf = concat_map_transf[:, :, :3]
            weightmap_transf = concat_map_transf[:, :, 3]
            if (label.shape[2] == 1):
                label_transf = concat_map_transf[:, :, -1:]
                label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
            else:
                label_transf = concat_map_transf[:, :, -3:]
            image_PIL = Image.fromarray(image_transf.astype(np.uint8))
            weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
            label_PIL = Image.fromarray(label_transf.astype(np.uint8))

            pics.append(image_PIL)
            pics.append(weightmap_PIL)
            pics.append(label_PIL)

        else:
            img = np.array(imgs[0])

        return tuple(pics)


class RandomChooseAug(object):
    def __call__(self, imgs):

        pics = []

        p_value = random.random()

        if p_value < 0.25:  # 0.25
            pics.append(imgs[0].filter(ImageFilter.BLUR))
            for k in range(1, len(imgs)):
                pics.append(imgs[k])
            return tuple(pics)

        elif p_value < 0.5:  # 0.5
            pics.append(imgs[0].filter(ImageFilter.GaussianBlur))
            for k in range(1, len(imgs)):
                pics.append(imgs[k])
            return tuple(pics)

        elif p_value < 0.75:  # 0.75
            pics.append(imgs[0].filter(ImageFilter.MedianFilter))
            for k in range(1, len(imgs)):
                pics.append(imgs[k])
            return tuple(pics)

        else:
            return imgs


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0, fill_val=(0,)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.fill_val = fill_val

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        pics = []

        w, h = imgs[0].size
        th, tw = self.size
        if (th > h or tw > w):
            ow = tw
            oh = th

            do_albu = 1
            if (do_albu == 1):
                transf = albu.Resize(always_apply=False, p=1.0, height=oh, width=ow, interpolation=0)
                image = np.array(imgs[0])
                weightmap = np.expand_dims(imgs[1], axis=2)
                label = np.array(imgs[2])  # np.expand_dims(imgs[2], axis=2)
                if (len(label.shape) == 2):
                    label = label.reshape(label.shape[0], label.shape[1], 1)
                if (len(image.shape) == 2):
                    image = image.reshape(image.shape[0], image.shape[1], 1)

                image_h, image_w = image.shape[:2]
                weightmap_h, weightmap_w = weightmap.shape[:2]
                label_h, label_w = label.shape[:2]

                if (
                        image_h != weightmap_h or image_h != label_h or image_w != weightmap_w or image_w != label_w or weightmap_h != label_h or weightmap_w != label_w):

                    image_transf = np.resize(image, (th, tw, 3))
                    weightmap_transf = np.resize(weightmap, (th, tw))
                    label_transf = np.resize(label, (th, tw, 3))
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))
                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)
                else:
                    concat_map = np.concatenate((image, weightmap, label), axis=2)

                    concat_map_transf = transf(image=np.array(concat_map))['image']
                    image_channel = image.shape[-1]
                    image_transf = concat_map_transf[:, :, :image_channel]
                    image_transf = np.squeeze(image_transf)
                    weightmap_transf = concat_map_transf[:, :, image_channel]
                    if (label.shape[2] == 1):
                        # label = label.reshape(label.shape[0], label.shape[1], 1)
                        label_transf = concat_map_transf[:, :, -1:]
                        label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
                    else:
                        label_transf = concat_map_transf[:, :, -3:]
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))

                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)
        else:
            do_albu = 1
            if (do_albu == 1):
                min_max_height = (int(th * 0.6), th)
                transf = albu.RandomSizedCrop(always_apply=False, p=1.0, min_max_height=min_max_height, height=th,
                                              width=tw,
                                              w2h_ratio=1.0, interpolation=0)

                image = np.array(imgs[0])
                weightmap = np.expand_dims(imgs[1], axis=2)
                label = np.array(imgs[2])  # np.expand_dims(imgs[2], axis=2)
                if (len(label.shape) == 2):
                    label = label.reshape(label.shape[0], label.shape[1], 1)
                if (len(image.shape) == 2):
                    image = image.reshape(image.shape[0], image.shape[1], 1)

                image_h, image_w = image.shape[:2]
                weightmap_h, weightmap_w = weightmap.shape[:2]
                label_h, label_w = label.shape[:2]

                if (
                        image_h != weightmap_h or image_h != label_h or image_w != weightmap_w or image_w != label_w or weightmap_h != label_h or weightmap_w != label_w):

                    image_transf = np.resize(image, (th, tw, 3))
                    weightmap_transf = np.resize(weightmap, (th, tw))
                    label_transf = np.resize(label, (th, tw, 3))

                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))

                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)

                else:

                    concat_map = np.concatenate((image, weightmap, label), axis=2)

                    concat_map_transf = transf(image=concat_map)['image']
                    image_channel = image.shape[-1]
                    image_transf = concat_map_transf[:, :, :image_channel]
                    image_transf = np.squeeze(image_transf)
                    weightmap_transf = concat_map_transf[:, :, image_channel]
                    if (label.shape[2] == 1):
                        label_transf = concat_map_transf[:, :, -1:]
                        label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
                    else:
                        label_transf = concat_map_transf[:, :, -3:]
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))

                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)



            else:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                for k in range(len(imgs)):
                    img = imgs[k]
                    if self.padding > 0:
                        img = ImageOps.expand(img, border=self.padding, fill=self.fill_val[k])

                    if w == tw and h == th:
                        pics.append(img)
                        continue

                    pics.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return tuple(pics)


class Sobel:
    _caches = {}
    ksize = 11

    @staticmethod
    def _generate_sobel_kernel(shape, axis):
        """
        shape must be odd: eg. (5,5)
        axis is the direction, with 0 to positive x and 1 to positive y
        """
        k = np.zeros(shape, dtype=float)
        p = [
            (j, i)
            for j in range(shape[0])
            for i in range(shape[1])
            if not (i == (shape[1] - 1) / 2.0 and j == (shape[0] - 1) / 2.0)
        ]

        for j, i in p:
            j_ = int(j - (shape[0] - 1) / 2.0)
            i_ = int(i - (shape[1] - 1) / 2.0)
            k[j, i] = (i_ if axis == 0 else j_) / float(i_ * i_ + j_ * j_)

        # return torch.from_numpy(k).unsqueeze(0)
        return ops.ExpandDims()(Tensor(k, mstype.float32), 0)

    @classmethod
    def kernel(cls, ksize=None):
        if ksize is None:
            ksize = cls.ksize
        if ksize in cls._caches:
            return cls._caches[ksize]

        sobel_x, sobel_y = (cls._generate_sobel_kernel((ksize, ksize), i) for i in (0, 1))

        # sobel_ker = torch.cat([sobel_y, sobel_x], dim=0).view(2, 1, ksize, ksize)
        sobel_ker = ops.Concat(0)([sobel_y, sobel_x]).view(2, 1, ksize, ksize)

        cls._caches[ksize] = sobel_ker
        return sobel_ker


ori_scales = {
    4: 1,
    8: 1,
    16: 2,
    32: 4,
}


class DTOffsetConfig:
    # energy configurations
    energy_level_step = int(os.environ.get('dt_energy_level_step', 5))
    assert energy_level_step > 0

    max_distance = int(os.environ.get('dt_max_distance', 5))
    min_distance = int(os.environ.get('dt_min_distance', 0))

    num_energy_levels = max_distance // energy_level_step + 1

    offset_min_level = int(os.environ.get('dt_offset_min_level', 0))
    offset_max_level = int(os.environ.get('dt_offset_max_level', 5))
    # assert 0 <= offset_min_level < num_energy_levels - 1
    # assert 0 < offset_max_level <= num_energy_levels

    # direction configurations
    direction_classes = 8
    num_classes = int(os.environ.get('dt_num_classes', direction_classes))  # 8 4 16
    assert num_classes in (4, 8, 16, 32,)

    # offset scale configurations
    scale = int(os.environ.get('dt_scale', ori_scales[num_classes]))
    assert scale % ori_scales[num_classes] == 0
    scale //= ori_scales[num_classes]

    c4_align_axis = os.environ.get('c4_align_axis') is not None


label_to_vector_mapping = {
    4: [
        [-1, -1], [-1, 1], [1, 1], [1, -1]
    ] if not DTOffsetConfig.c4_align_axis else [
        [0, -1], [-1, 0], [0, 1], [1, 0]
    ],
    5: [
        [0, 0], [-1, -1], [-1, 1], [1, 1], [1, -1]
    ] if not DTOffsetConfig.c4_align_axis else [
        [0, -1], [-1, 0], [0, 1], [1, 0]
    ],
    8: [
        [0, -1], [-1, -1], [-1, 0], [-1, 1],
        [0, 1], [1, 1], [1, 0], [1, -1]
    ],
    9: [
        [0, 0], [0, -1], [-1, -1], [-1, 0], [-1, 1],
        [0, 1], [1, 1], [1, 0], [1, -1]
    ],
    16: [
        [0, -2], [-1, -2], [-2, -2], [-2, -1],
        [-2, 0], [-2, 1], [-2, 2], [-1, 2],
        [0, 2], [1, 2], [2, 2], [2, 1],
        [2, 0], [2, -1], [2, -2], [1, -2]
    ],
    17: [
        [0, 0], [0, -2], [-1, -2], [-2, -2], [-2, -1],
        [-2, 0], [-2, 1], [-2, 2], [-1, 2],
        [0, 2], [1, 2], [2, 2], [2, 1],
        [2, 0], [2, -1], [2, -2], [1, -2]
    ],

    32: [
        [0, -4], [-1, -4], [-2, -4], [-3, -4], [-4, -4], [-4, -3], [-4, -2], [-4, -1],
        [-4, 0], [-4, 1], [-4, 2], [-4, 3], [-4, 4], [-3, 4], [-2, 4], [-1, 4],
        [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [4, 3], [4, 2], [4, 1],
        [4, 0], [4, -1], [4, -2], [4, -3], [4, -4], [3, -4], [2, -4], [1, -4],
    ]
}

vector_to_label_mapping = {
    8: list(range(8)),
    16: list(range(16)),
}


class DTOffsetHelper:

    @staticmethod
    def encode_multi_labels(dir_labels):
        """
        Only accept ndarray of shape H x W (uint8).
        """
        assert isinstance(dir_labels, np.ndarray)

        output = np.zeros((*dir_labels.shape, 8), dtype=np.int)
        for i in range(8):
            output[..., i] = (dir_labels & (1 << i) != 0).astype(np.int)

        return output

    @staticmethod
    def get_opposite_angle(angle_map):
        new_angle_map = angle_map + 180
        mask = (new_angle_map >= 180) & (new_angle_map <= 360)
        new_angle_map[mask] = new_angle_map[mask] - 360
        return new_angle_map

    @staticmethod
    def angle_to_vector(angle_map,
                        num_classes=DTOffsetConfig.num_classes,
                        return_tensor=False):

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
        else:
            assert isinstance(angle_map, np.ndarray)

        if return_tensor:
            lib = torch
            vector_map = torch.zeros((*angle_map.shape, 2), dtype=torch.float).to(angle_map.device)
            deg2rad = lambda x: np.pi / 180.0 * x
        else:
            lib = np
            vector_map = np.zeros((*angle_map.shape, 2), dtype=np.float)
            deg2rad = np.deg2rad

        if num_classes is not None:
            angle_map, _ = DTOffsetHelper.align_angle(angle_map, num_classes=num_classes, return_tensor=return_tensor)

        angle_map = deg2rad(angle_map)

        vector_map[..., 0] = lib.sin(angle_map)
        vector_map[..., 1] = lib.cos(angle_map)

        return vector_map

    @staticmethod
    def align_angle(angle_map,
                    num_classes=DTOffsetConfig.num_classes,
                    return_tensor=False):
        # print(angle_map.max(), angle_map.min())
        if num_classes == 4 and not DTOffsetConfig.c4_align_axis:
            return DTOffsetHelper.align_angle_c4(angle_map, return_tensor=return_tensor)

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
        else:
            assert isinstance(angle_map, np.ndarray)

        step = 360 / num_classes
        if return_tensor:
            new_angle_map = torch.zeros(angle_map.shape).float().to(angle_map.device)
            angle_index_map = torch.zeros(angle_map.shape).long().to(angle_map.device)
        else:
            new_angle_map = np.zeros(angle_map.shape, dtype=np.float)
            angle_index_map = np.zeros(angle_map.shape, dtype=np.int)
        mask = (angle_map <= (-180 + step / 2)) | (angle_map > (180 - step / 2))
        new_angle_map[mask] = -180
        angle_index_map[mask] = 0

        for i in range(1, num_classes):
            middle = -180 + step * i
            mask = (angle_map > (middle - step / 2)) & (angle_map <= (middle + step / 2))
            new_angle_map[mask] = middle
            angle_index_map[mask] = i

        return new_angle_map, angle_index_map

    @staticmethod
    def vector_to_label(vector_map,
                        num_classes=DTOffsetConfig.num_classes,
                        return_tensor=False):

        if return_tensor:
            assert isinstance(vector_map, torch.Tensor)
        else:
            assert isinstance(vector_map, np.ndarray)

        if return_tensor:
            rad2deg = lambda x: x * 180. / np.pi
        else:
            rad2deg = np.rad2deg

        angle_map = np.arctan2(vector_map[..., 0], vector_map[..., 1])
        angle_map = rad2deg(angle_map)

        return DTOffsetHelper.angle_to_direction_label(angle_map,
                                                       return_tensor=return_tensor,
                                                       num_classes=num_classes)

    @staticmethod
    def label_to_vector(labelmap, num_classes=DTOffsetConfig.num_classes):
        mapping = label_to_vector_mapping[num_classes]
        offset_h = ops.ZerosLike()(labelmap).asnumpy()
        offset_w = ops.ZerosLike()(labelmap).asnumpy()

        labelmap = labelmap.asnumpy()

        for idx, (hdir, wdir) in enumerate(mapping):
            mask = labelmap == idx
            offset_h[mask] = hdir
            offset_w[mask] = wdir

        offset_h = Tensor(offset_h)
        offset_w = Tensor(offset_w)
        ret = ops.Stack(axis=-1)([offset_h, offset_w])
        ret = ret.transpose(0, 3, 1, 2)

        return ret

    @staticmethod
    def angle_to_direction_label(angle_map,
                                 seg_label_map=None,
                                 distance_map=None,
                                 num_classes=DTOffsetConfig.num_classes,
                                 extra_ignore_mask=None,
                                 return_tensor=False):
        # print('DTOffsetConfig.num_classes:{}'.format(DTOffsetConfig.num_classes))
        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
            assert isinstance(seg_label_map, torch.Tensor) or seg_label_map is None
        else:
            assert isinstance(angle_map, np.ndarray)
            assert isinstance(seg_label_map, np.ndarray) or seg_label_map is None

        _, label_map = DTOffsetHelper.align_angle(angle_map,
                                                  num_classes=num_classes,
                                                  return_tensor=return_tensor)
        if distance_map is not None:
            label_map[distance_map > DTOffsetConfig.max_distance] = num_classes
        if seg_label_map is None:
            if return_tensor:
                ignore_mask = torch.zeros(angle_map.shape, dtype=torch.uint8).to(angle_map.device)
            else:
                ignore_mask = np.zeros(angle_map.shape, dtype=np.bool)
        else:
            ignore_mask = seg_label_map == -1

        if extra_ignore_mask is not None:
            ignore_mask = ignore_mask | extra_ignore_mask
        label_map[ignore_mask] = -1

        return label_map


def get_centerpoint2(mask, n, m):
    # print(dis.shape)
    now = -1
    x = -1
    y = -1
    P = []
    for i in range(8):
        P.append((math.sin(2 * math.pi / 8 * i), math.cos(2 * math.pi / 8 * i)))
    for i in range(n):
        for j in range(m):
            if mask[i][j] > 0:

                ma = 0
                mi = 10000000
                for k in range(8):
                    l = 0
                    r = 1000
                    for tim in range(30):
                        mid = (l + r) / 2
                        nx = round(i + P[k][0] * mid)
                        ny = round(j + P[k][1] * mid)
                        if (nx >= 0 and nx < n and ny >= 0 and ny < m and mask[nx][ny] > 0):
                            l = mid
                        else:
                            r = mid
                    ma = max(ma, r)
                    mi = min(mi, r)
                assert (ma > 0 and mi > 0)
                centerness = mi / ma
                if centerness > now:
                    now = centerness
                    x = i
                    y = j
    return [int(x), int(y)]


class LabelEncoding(object):
    """
    Encoding the label, computes boundary individually
    """

    def __init__(self, out_c=3, radius=1, do_direction=0):
        self.out_c = out_c
        self.radius = 1  # radius
        self.do_direction = do_direction

    def __call__(self, imgs):
        start_time = time.time()
        time_str = str(start_time)[-5:]

        out_imgs = list(imgs)
        label = imgs[2]  # imgs[-1]

        if not isinstance(label, np.ndarray):
            label = np.array(label)

        min_value = 190
        max_value = 210
        half_value = 255 * 0.5  # 0

        # if unique>2，input = instance level
        if (len(label.shape) == 2):
            label_inside = label
            label_level_len = len(np.unique(label))
        else:
            label_inside = label[:, :, 0]
            label_level_len = len(np.unique(label_inside))

        if (self.out_c != 3):
            if (label_level_len > 2):
                ins3channel = measure.label(label)
                label_instance = ins3channel[:, :, 0]
                label_instance = measure.label(label_instance)
                new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                new_label[label_instance > 0] = 2  # inside
                new_label_inside = copy.deepcopy(new_label)
                # boun_instance = morphology.dilation(label_instance) & (~morphology.erosion(label_instance, morphology.disk(self.radius)))
                # new_label[boun_instance > 0] = 2
            else:
                new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                new_label[label[:, :, 0] > half_value] = 2  # inside
                new_label[label[:, :, 1] > half_value] = 2  # inside
                new_label = morphology.erosion(new_label, morphology.disk(self.radius))
                new_label_inside = copy.deepcopy(new_label)
                label_instance = measure.label(new_label_inside)
                # boun = morphology.dilation(new_label) & (~morphology.erosion(new_label, morphology.disk(self.radius)))
                # new_label[boun > 0] = 2  # boundary

        else:
            # if label_level_len>2，input = instance level
            if (label_level_len > 2):
                new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                new_label[label_inside > 0] = 1  # inside
                new_label = remove_small_objects(new_label, 5)

                new_label_inside = copy.deepcopy(new_label)

                boun_instance = morphology.dilation(label_inside) & (
                    ~morphology.erosion(label_inside, morphology.disk(self.radius)))
                new_label[boun_instance > 0] = 2
                postproc = 1
                if (postproc == 0):
                    label_inside_new = (new_label == 1).astype(np.uint8)
                    label_instance = measure.label(label_inside_new)
                    label_instance = morphology.dilation(label_instance, selem=morphology.selem.disk(self.radius))
                else:
                    label_inside_new = (new_label == 1).astype(np.uint8)
                    label_instance = postproc_other.process(label_inside_new.astype(np.uint8) * 255,
                                                            model_mode='modelName', min_size=5)
                    label_instance = morphology.dilation(label_instance, selem=morphology.selem.disk(self.radius))


            else:
                # print('输入的是3分类 label')
                new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                new_label[label_inside > half_value] = 1  # inside
                new_label_inside = copy.deepcopy(new_label)
                boun = morphology.dilation(new_label) & (~morphology.erosion(new_label, morphology.disk(self.radius)))
                new_label[boun > 0] = 2  # boundary
                postproc = 0
                if (postproc == 0):
                    label_inside_new = (new_label == 1).astype(np.uint8)
                    label_instance = measure.label(label_inside_new)
                    label_instance = morphology.dilation(label_instance, selem=morphology.selem.disk(self.radius))

                else:
                    label_inside_new = (new_label == 1).astype(np.uint8)
                    label_instance = postproc_other.process(label_inside_new.astype(np.uint8) * 255,
                                                            model_mode='modelName', min_size=5)
                    label_instance = morphology.dilation(label_instance, selem=morphology.selem.disk(self.radius))

        label1 = Image.fromarray((new_label / 2 * 255).astype(np.uint8))
        out_imgs[2] = label1

        do_direction = self.do_direction
        if (do_direction == 1):
            height, width = label.shape[0], label.shape[1]
            distance_map = np.zeros((height, width), dtype=np.float)
            distance_center_map = np.zeros((height, width), dtype=np.float)

            dir_map = np.zeros((height, width, 2), dtype=np.float32)
            ksize = 11
            point_number = 0
            label_point = np.zeros((height, width), dtype=np.float)

            mask = label_instance
            markers_unique = np.unique(label_instance)
            markers_len = len(np.unique(label_instance)) - 1

            for k in markers_unique[1:]:
                nucleus = (mask == k).astype(np.int)
                distance_i = distance_transform_edt(nucleus)
                distance_i_normal = distance_i / distance_i.max()
                distance_map = distance_map + distance_i_normal

                # local_maxi = feature.peak_local_max(distance_i, exclude_border=0, num_peaks=1)
                # if (local_maxi.shape[0] != 1):
                #    print(local_maxi)
                # assert local_maxi.shape[0] > 0
                # assert nucleus[local_maxi[0][0], local_maxi[0][1]] > 0
                # label_point[local_maxi[0][0], local_maxi[0][1]] = 255.0

                center = get_centerpoint2(nucleus, nucleus.shape[0], nucleus.shape[1])
                local_maxi = [center]
                assert nucleus[center[0], center[1]] > 0
                label_point[center[0], center[1]] = 255.0

                # if (do_direction == 1):
                nucleus = morphology.dilation(nucleus, morphology.disk(self.radius))
                point_map_k = np.zeros((height, width), dtype=np.int)
                point_map_k[local_maxi[0][0], local_maxi[0][1]] = 1
                int_pos = distance_transform_edt(1 - point_map_k)
                int_pos = int_pos * nucleus
                distance_center_i = (1 - int_pos / (int_pos.max() + 0.0000001)) * nucleus
                distance_center_map = distance_center_map + distance_center_i

                dir_i = np.zeros_like(dir_map)
                sobel_kernel = Sobel.kernel(ksize=ksize)

                # print(distance_center_i.shape, height, width, sobel_kernel.shape, sobel_kernel.dtype)
                # (256, 256) 256 256 (2, 1, 11, 11) 11
                # dir_i = torch.nn.functional.conv2d(
                #     torch.from_numpy(distance_center_i).float().view(1, 1, height, width),
                #     sobel_kernel, padding=ksize // 2).squeeze().permute(1, 2, 0).numpy()

                __input = Tensor(distance_center_i, dtype=sobel_kernel.dtype).view(1, 1, height, width)
                conv2d = ops.Conv2D(out_channel=sobel_kernel.shape[0], kernel_size=ksize, pad_mode="pad",
                                    pad=ksize // 2)
                dir_i = ops.Transpose()(conv2d(__input, sobel_kernel).squeeze(), (1, 2, 0)).asnumpy()

                dir_i[(nucleus == 0), :] = 0
                dir_map[(nucleus != 0), :] = 0
                dir_map += dir_i
                point_number = point_number + 1
            assert int(label_point.sum() / 255) == markers_len

            t_time = time.time()

            distance_map = distance_center_map

            label_point_gaussian = ndimage.gaussian_filter(label_point, sigma=2, order=0).astype(np.float16)
            out_imgs.append(label_point_gaussian)

            # 角度
            angle = np.degrees(np.arctan2(dir_map[:, :, 0], dir_map[:, :, 1]))
            label_angle = copy.deepcopy(angle)
            label_angle[new_label_inside == 0] = -180
            label_angle = label_angle + 180
            angle[new_label_inside == 0] = 0
            vector = (DTOffsetHelper.angle_to_vector(angle, return_tensor=False))
            # direction class
            label_direction = DTOffsetHelper.vector_to_label(vector, return_tensor=False)
            label_direction_new = copy.deepcopy(label_direction)
            # input = instance level
            if (label_level_len > 2):
                label_direction_new[new_label_inside == 0] = -1
            # input = 3-class level

            else:
                # label_direction_new[new_label == 0] = -1
                label_direction_new[new_label_inside == 0] = -1
            label_direction_new2 = label_direction_new + 1

            direction_label = True  # True False
            if (direction_label == False):
                out_imgs.append(label_angle)
            else:
                out_imgs.append(label_direction_new2)




        else:
            min_value = 190
            # label_point_gaussian = np.zeros((height, width), dtype=np.float)
            # label_direction_new = np.zeros((height, width), dtype=np.float)
            # out_imgs.append(label_point_gaussian)
            # out_imgs.append(label_direction_new)

        return tuple(out_imgs)


selector = {
    'random_color': lambda x: RandomColor(x),  # change to later
    'horizontal_flip': lambda x: RandomHorizontalFlip(),
    'vertical_flip': lambda x: RandomVerticalFlip(),
    'random_elastic': lambda x: RandomElasticDeform(x[0], x[1]),
    'random_chooseAug': lambda x: RandomChooseAug(),
    'random_crop': lambda x: RandomCrop(x),
    'label_encoding': lambda x: LabelEncoding(x[0], x[1], x[2])
}


def get_transforms_list(transform_dict):
    # transform_dict = OrderedDict([('random_color', 1), ('horizontal_flip', True), ('vertical_flip', True),
    #                               ('random_elastic', [6, 15]), ('random_chooseAug', 1), ('random_crop', 256),
    #                               ('label_encoding', [3, 2, 1])])
    transform_list = []
    for k, v in transform_dict.items():
        transform_list.append(selector[k](v))

    return transform_list
