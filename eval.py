# -*- coding: utf-8 -*-
"""
@author: huangxs
@License: (C)Copyright 2021, huangxs
@CreateTime: 2021/11/16 19:10:00
@Filename: eval

"""
import os

# 设置临时环境变量，只输出error日志
os.environ['GLOG_v'] = "3"
os.environ['DEVICE_ID'] = "5"

import numpy as np
from skimage import io
from skimage import measure
import skimage.morphology as morph
import copy

from PIL import Image
from src.utils.metrics_util import accuracy_pixel_level
from collections import OrderedDict

import mindspore.dataset as ds
from sklearn.metrics import jaccard_score
from src.cdnet import CDNet
from src.utils.dataset import MoNuSegGenerator, MoNuSegPreparedGenerator
from src.utils.direction_transform import get_transforms_list, DTOffsetHelper
from src.utils.loss import *

import glob
import numpy as np
import time
from scipy import ndimage as ndi

import mindspore.dataset.vision.py_transforms as py_vision
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


def run_eval():
    # ====== model ======
    _cdnet = CDNet(backbone_name='vgg16_bn', encoder_freeze=False, classes=3)

    # checkpoint file
    checkpoint_path = 'checkpoint/cdnet_eval.ckpt'
    if len(checkpoint_path) > 0:
        print('load checkpoint:', checkpoint_path)
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(_cdnet, param_dict)

    # 获取测试图片数据
    image_dir = 'data/MoNuSeg_oridata/images/test1'
    label_dir = 'data/MoNuSeg_oridata/labels/test1'
    annotation_dir = 'data/MoNuSeg_oridata/Annotations/test1'

    image_path_list = glob.glob(os.path.join(image_dir, '*.png'))

    count_pred_list = []
    count_label_list = []
    ji_value = 0
    counter = 0
    all_hover_AJI = 0.0
    all_hover_Dice = 0.0

    for _index, image_path in enumerate(image_path_list):
        base_name = os.path.basename(image_path).replace('.png', '')

        # 转换图片到tensor，image,label
        _image = Image.open(image_path).convert('RGB')

        ori_h = _image.size[1]
        ori_w = _image.size[0]

        eval_flag = True
        if eval_flag:
            label_path = os.path.join(label_dir, '%s_label.png' % base_name)
            annotation_path = os.path.join(annotation_dir, '%s.xml' % base_name)

            label_instance_path = '{:s}_ins/{:s}.npy'.format(label_dir, base_name)
            label_img_instance = np.load(label_instance_path)
            # print('{}, label_img_instance.len = {}'.format(label_instance_path, len(np.unique(label_img_instance))))

            label_img = io.imread(label_path)
            multiple_number = 255

        label_ins_h = label_img_instance.shape[0]
        label_ins_w = label_img_instance.shape[1]

        _input = Tensor(py_vision.ToTensor()(_image)).view((1, 3, _image.size[0], _image.size[1]))

        # prob_maps, point_maps, prob_dcm
        prob_run_time = 0
        probmap_list = get_probmaps(_input, _cdnet, prob_run_time)
        prob_maps = probmap_list[0]
        # print('len(probmap_list) = {}'.format(len(probmap_list)))

        point_maps = probmap_list[1]
        prob_dcm = probmap_list[2]

        tta = True
        if tta:
            img_hf = _image.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_vf = _image.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_hvf = img_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips
            input_hf = Tensor(py_vision.ToTensor()(img_hf)).view((1, 3, img_hf.size[0], img_hf.size[1]))
            input_vf = Tensor(py_vision.ToTensor()(img_vf)).view((1, 3, img_vf.size[0], img_vf.size[1]))
            input_hvf = Tensor(py_vision.ToTensor()(img_hvf)).view((1, 3, img_hvf.size[0], img_hvf.size[1]))

            prob_maps_hf_list = get_probmaps(input_hf, _cdnet, prob_run_time)
            prob_maps_vf_list = get_probmaps(input_vf, _cdnet, prob_run_time)
            prob_maps_hvf_list = get_probmaps(input_hvf, _cdnet, prob_run_time)

            prob_maps_hf = prob_maps_hf_list[0]

            point_maps_hf = prob_maps_hf_list[1]
            prob_dcm_hf = prob_maps_hf_list[2]

            prob_maps_vf = prob_maps_vf_list[0]

            point_maps_vf = prob_maps_vf_list[1]
            prob_dcm_vf = prob_maps_vf_list[2]

            prob_maps_hvf = prob_maps_hvf_list[0]
            point_maps_hvf = prob_maps_hvf_list[1]
            prob_dcm_hvf = prob_maps_hvf_list[2]

            # re flip
            prob_maps_hf = np.flip(prob_maps_hf, 2)
            prob_maps_vf = np.flip(prob_maps_vf, 1)
            prob_maps_hvf = np.flip(np.flip(prob_maps_hvf, 1), 2)

            point_maps_hf = np.flip(point_maps_hf, 2)
            point_maps_vf = np.flip(point_maps_vf, 1)
            point_maps_hvf = np.flip(np.flip(point_maps_hvf, 1), 2)

            prob_dcm_hf = np.flip(prob_dcm_hf, 2)
            prob_dcm_vf = np.flip(prob_dcm_vf, 1)
            prob_dcm_hvf = np.flip(np.flip(prob_dcm_hvf, 1), 2)

            # rotation 90 and flips
            img_r90 = _image.rotate(90, expand=True)
            img_r90_hf = img_r90.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_r90_vf = img_r90.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_r90_hvf = img_r90_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips
            input_r90 = Tensor(py_vision.ToTensor()(img_r90)).view((1, 3, img_r90.size[0], img_r90.size[1]))
            input_r90_hf = Tensor(py_vision.ToTensor()(img_r90_hf)).view((1, 3, img_r90_hf.size[0], img_r90_hf.size[1]))
            input_r90_vf = Tensor(py_vision.ToTensor()(img_r90_vf)).view((1, 3, img_r90_vf.size[0], img_r90_vf.size[1]))
            input_r90_hvf = Tensor(py_vision.ToTensor()(img_r90_hvf)).view(
                (1, 3, img_r90_hvf.size[0], img_r90_hvf.size[1]))

            prob_maps_r90_list = get_probmaps(input_r90, _cdnet, prob_run_time)
            prob_maps_r90_hf_list = get_probmaps(input_r90_hf, _cdnet, prob_run_time)
            prob_maps_r90_vf_list = get_probmaps(input_r90_vf, _cdnet, prob_run_time)
            prob_maps_r90_hvf_list = get_probmaps(input_r90_hvf, _cdnet, prob_run_time)

            prob_maps_r90 = prob_maps_r90_list[0]
            point_maps_r90 = prob_maps_r90_list[1]
            prob_dcm_r90 = prob_maps_r90_list[2]

            prob_maps_r90_hf = prob_maps_r90_hf_list[0]
            point_maps_r90_hf = prob_maps_r90_hf_list[1]
            prob_dcm_r90_hf = prob_maps_r90_hf_list[2]

            prob_maps_r90_vf = prob_maps_r90_vf_list[0]
            point_maps_r90_vf = prob_maps_r90_vf_list[1]
            prob_dcm_r90_vf = prob_maps_r90_vf_list[2]

            prob_maps_r90_hvf = prob_maps_r90_hvf_list[0]
            point_maps_r90_hvf = prob_maps_r90_hvf_list[1]
            prob_dcm_r90_hvf = prob_maps_r90_hvf_list[2]

            # re flip
            prob_maps_r90 = np.rot90(prob_maps_r90, k=3, axes=(1, 2))
            prob_maps_r90_hf = np.rot90(np.flip(prob_maps_r90_hf, 2), k=3, axes=(1, 2))
            prob_maps_r90_vf = np.rot90(np.flip(prob_maps_r90_vf, 1), k=3, axes=(1, 2))
            prob_maps_r90_hvf = np.rot90(np.flip(np.flip(prob_maps_r90_hvf, 1), 2), k=3, axes=(1, 2))

            point_maps_r90 = np.rot90(point_maps_r90, k=3, axes=(1, 2))
            point_maps_r90_hf = np.rot90(np.flip(point_maps_r90_hf, 2), k=3, axes=(1, 2))
            point_maps_r90_vf = np.rot90(np.flip(point_maps_r90_vf, 1), k=3, axes=(1, 2))
            point_maps_r90_hvf = np.rot90(np.flip(np.flip(point_maps_r90_hvf, 1), 2), k=3, axes=(1, 2))

            prob_dcm_r90 = np.rot90(prob_dcm_r90, k=3, axes=(1, 2))
            prob_dcm_r90_hf = np.rot90(np.flip(prob_dcm_r90_hf, 2), k=3, axes=(1, 2))
            prob_dcm_r90_vf = np.rot90(np.flip(prob_dcm_r90_vf, 1), k=3, axes=(1, 2))
            prob_dcm_r90_hvf = np.rot90(np.flip(np.flip(prob_dcm_r90_hvf, 1), 2), k=3, axes=(1, 2))

            prob_maps = (prob_maps + prob_maps_hf + prob_maps_vf + prob_maps_hvf
                         + prob_maps_r90 + prob_maps_r90_hf + prob_maps_r90_vf + prob_maps_r90_hvf) / 8

            point_maps = (point_maps + point_maps_hf + point_maps_vf + point_maps_hvf
                          + point_maps_r90 + point_maps_r90_hf + point_maps_r90_vf + point_maps_r90_hvf) / 8

        dcm_combined = 1
        if (dcm_combined == 1):
            # print('====> dcm_combined  =======')
            # ins_label_fromd [1,h,w]
            prob_dcm_map = np.zeros((prob_dcm.shape[1], prob_dcm.shape[2], 8), np.uint8)
            prob_dcm_map[:, :, 0] = prob_dcm[0]
            prob_dcm_map[:, :, 1] = prob_dcm_hf[0]
            prob_dcm_map[:, :, 2] = prob_dcm_vf[0]
            prob_dcm_map[:, :, 3] = prob_dcm_hvf[0]
            prob_dcm_map[:, :, 4] = prob_dcm_r90[0]
            prob_dcm_map[:, :, 5] = prob_dcm_r90_hf[0]
            prob_dcm_map[:, :, 6] = prob_dcm_r90_vf[0]
            prob_dcm_map[:, :, 7] = prob_dcm_r90_hvf[0]

            pred_dcm = prob_dcm[0]

            voting_firt = 0
            if (voting_firt == 1):
                pass
            else:
                prob_ddm_map = np.zeros((prob_dcm.shape[1], prob_dcm.shape[2], 8), float)
                prob_ddm_map[:, :, 0] = generate_dd_map(prob_dcm_map[:, :, 0], 9)
                prob_ddm_map[:, :, 1] = generate_dd_map(prob_dcm_map[:, :, 1], 9)
                prob_ddm_map[:, :, 2] = generate_dd_map(prob_dcm_map[:, :, 2], 9)
                prob_ddm_map[:, :, 3] = generate_dd_map(prob_dcm_map[:, :, 3], 9)
                prob_ddm_map[:, :, 4] = generate_dd_map(prob_dcm_map[:, :, 4], 9)
                prob_ddm_map[:, :, 5] = generate_dd_map(prob_dcm_map[:, :, 5], 9)
                prob_ddm_map[:, :, 6] = generate_dd_map(prob_dcm_map[:, :, 6], 9)
                prob_ddm_map[:, :, 7] = generate_dd_map(prob_dcm_map[:, :, 7], 9)

                pred_direction = np.mean(prob_ddm_map, axis=2)
                prob_direction_maps = pred_direction.reshape(1, pred_direction.shape[0], pred_direction.shape[1])

                branch0 = 5
                # filename = '{:s}/b{:s}_{:s}_pred_direction_combined0.png'.format(seg_folder, str(branch0), name)
                # cv2.imwrite(filename, prob_ddm_map[:, :, 0] * 255)
            branch0 = 5
            # filename = '{:s}/b{:s}_{:s}_pred_direction_combined.png'.format(seg_folder, str(branch0), name)
            # cv2.imwrite(filename, prob_direction_maps[0] * 255)

        point_maps_max = np.max(point_maps[0, :, :])
        point_maps_min = np.min(point_maps[0, :, :])
        point_maps_normal = (point_maps[0, :, :] - point_maps_min) / (point_maps_max - point_maps_min)
        # io.imsave('{:s}/Point_{:s}.png'.format(seg_folder, name), point_maps[0, :, :])  # point_maps_normal * 255.0
        predicted_counts = np.sum(point_maps) / multiple_number
        real_counts = len(np.unique(label_img_instance))
        # print(base_name, predicted_counts, real_counts)
        count_pred_list.append(predicted_counts)
        count_label_list.append(real_counts)

        DDM_switch = 100
        pred_inside3 = (point_maps[0] / np.max(point_maps) > 0.2) * 1
        pred_inside3 = morph.dilation(pred_inside3, selem=morph.selem.disk(1))
        prob_direction_map_pred_inside = prob_direction_maps[0] * pred_inside3
        enhanced_boundary = prob_direction_maps[0] - prob_direction_map_pred_inside
        enhanced_boundary = 2 * enhanced_boundary
        assert (np.min(enhanced_boundary) >= 0)
        prob_maps[2, :, :] = (prob_maps[2, :, :] + 0.5 * enhanced_boundary) * (1 + enhanced_boundary)
        pred = np.argmax(prob_maps, axis=0)
        pred_inside = pred == 1
        pred_foreground = pred > 0

        pred_inside2 = ndi.binary_fill_holes(pred_inside)

        pred2 = morph.remove_small_objects(pred_inside2, 20)  # remove small object

        pred2 = pred2.astype(np.uint8)
        pred_labeled = measure.label(pred2)  # connected component labeling
        pred_labeled = morph.dilation(pred_labeled, selem=morph.selem.disk(2))

        pred_labeled2 = pred2.astype(np.uint8) * 255

        label_inside = label_img[:, :] > 0
        label_instance_img = copy.deepcopy(label_img_instance)
        label_img = (label_img_instance[:, :] > 0).astype(np.uint8) * 255

        ji1 = jaccard_score(pred_labeled2, label_img, average='samples', zero_division=0.0)

        ji_value += ji1

        label_img = label_instance_img

        ##### not finish
        label_instance_img = copy.deepcopy(label_img_instance)
        label_img = label_instance_img
        gt_labeled = measure.label(label_img)

        pred_labeled = morph.dilation(pred_labeled, selem=morph.selem.disk(2))
        pred_labeled = measure.label(pred_labeled)

        result_AJI, analysis_FP, analysis_FN, _, _ = get_fast_aji(gt_labeled, pred_labeled)
        result_Dice = get_dice_1(gt_labeled, pred_labeled)

        all_hover_AJI += result_AJI
        all_hover_Dice += result_Dice
        counter += 1

        print('%d : [%s], AJI_sklearn:%.4f, AJI:%.4f, Dice:%.4f' % (_index, base_name, ji1, result_AJI, result_Dice))

    AJI_sklearn_mean = ji_value / counter
    hover_AJI = all_hover_AJI / counter
    hover_Dice = all_hover_Dice / counter
    print('AJI_sklearn:%.4f, hover_AJI:%.4f, hover_Dice:%.4f' % (AJI_sklearn_mean, hover_AJI, hover_Dice))
    return AJI_sklearn_mean


def get_probmaps(input, model, prob_run_times):
    size = 0  # 0 all_image

    output_all = model(input)

    _output = output_all[0]
    output_point = output_all[1]
    output_direction = output_all[2]  # [1, 10, 1000, 1000]
    _output = _output.squeeze(0)
    output_point = output_point.squeeze(0)
    output_direction = output_direction.squeeze(0)  # [10, 1000, 1000]
    # print('output_direction.shape={}'.format(output_direction.shape))

    point_maps = output_point.asnumpy()

    prob_maps = nn.Softmax(axis=0)(_output).asnumpy()

    direction_label = True  # True False

    pred_3c = np.argmax(prob_maps, axis=0)
    pred_inside = pred_3c != 0

    prob_maps_direction = nn.Softmax(axis=0)(output_direction).asnumpy()

    prob_maps_direction[0, :, :] = prob_maps_direction[0, :, :] * prob_maps[0, :, :]
    pred_direction = np.argmax(prob_maps_direction, axis=0)

    pred_direction = pred_direction.reshape(1, pred_direction.shape[0], pred_direction.shape[1])

    prob_run_times += 1

    return_list = [prob_maps, point_maps, pred_direction]

    return return_list


def get_fast_aji(true, pred):
    """
    AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    # 多检
    pairwise_FP = np.zeros([len(true_id_list) - 1,
                            len(pred_id_list) - 1], dtype=np.float64)
    # 漏检
    pairwise_FN = np.zeros([len(true_id_list) - 1,
                            len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

            pairwise_FP[true_id - 1, pred_id - 1] = p_mask.sum() - inter
            pairwise_FN[true_id - 1, pred_id - 1] = t_mask.sum() - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)

    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    overall_FP = (pairwise_FP[paired_true, paired_pred]).sum()
    overall_FN = (pairwise_FN[paired_true, paired_pred]).sum()

    #
    paired_true = (list(paired_true + 1))  # index to instance ID
    paired_pred = (list(paired_pred + 1))
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])

    less_pred = 0
    more_pred = 0

    for true_id in unpaired_true:
        less_pred += true_masks[true_id].sum()
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        more_pred += pred_masks[pred_id].sum()
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    fm = overall_union - overall_inter
    # print('\t [ana_FP = {:.4f}, ana_FN = {:.4f}, ana_less = {:.4f}, ana_more = {:.4f}]'.format((overall_FP / fm),
    #                                                                                            (overall_FN / fm),
    #                                                                                            (less_pred / fm),
    #                                                                                            (more_pred / fm)))

    return aji_score, overall_FP / fm, overall_FN / fm, less_pred / fm, more_pred / fm


def get_dice_1(true, pred):
    """
        Traditional dice
    """
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)


def circshift(matrix_ori, direction, shiftnum1, shiftnum2):
    # direction = 1,2,3,4 # 偏移方向 1:左上; 2:右上; 3:左下; 4:右下;
    c, h, w = matrix_ori.shape
    matrix_new = np.zeros_like(matrix_ori)

    for k in range(c):
        matrix = matrix_ori[k]
        # matrix = matrix_ori[:,:,k]
        if (direction == 1):
            # 左上
            matrix = np.vstack((matrix[shiftnum1:, :], np.zeros_like(matrix[:shiftnum1, :])))
            matrix = np.hstack((matrix[:, shiftnum2:], np.zeros_like(matrix[:, :shiftnum2])))
        elif (direction == 2):
            # 右上
            matrix = np.vstack((matrix[shiftnum1:, :], np.zeros_like(matrix[:shiftnum1, :])))
            matrix = np.hstack((np.zeros_like(matrix[:, (w - shiftnum2):]), matrix[:, :(w - shiftnum2)]))
        elif (direction == 3):
            # 左下
            matrix = np.vstack((np.zeros_like(matrix[(h - shiftnum1):, :]), matrix[:(h - shiftnum1), :]))
            matrix = np.hstack((matrix[:, shiftnum2:], np.zeros_like(matrix[:, :shiftnum2])))
        elif (direction == 4):
            # 右下
            matrix = np.vstack((np.zeros_like(matrix[(h - shiftnum1):, :]), matrix[:(h - shiftnum1), :]))
            matrix = np.hstack((np.zeros_like(matrix[:, (w - shiftnum2):]), matrix[:, :(w - shiftnum2)]))
        # matrix_new[k]==>matrix_new[:,:, k]
        # matrix_new[:,:, k] = matrix
        matrix_new[k] = matrix

    return matrix_new


def generate_dd_map(label_direction, direction_classes):
    direction_offsets = DTOffsetHelper.label_to_vector(
        Tensor(label_direction.reshape(1, label_direction.shape[0], label_direction.shape[1]), dtype=mstype.int32),
        direction_classes)
    direction_offsets = direction_offsets[0].transpose(1, 2, 0).asnumpy()

    direction_os = direction_offsets  # [256,256,2]

    height, weight = direction_os.shape[0], direction_os.shape[1]

    cos_sim_map = np.zeros((height, weight), dtype=float)

    feature_list = []
    feature5 = direction_os  # .transpose(1, 2, 0)
    if (direction_classes - 1 == 4):
        direction_os = direction_os.transpose(2, 0, 1)
        feature2 = circshift(direction_os, 1, 1, 0).transpose(1, 2, 0)
        feature4 = circshift(direction_os, 3, 0, 1).transpose(1, 2, 0)
        feature6 = circshift(direction_os, 4, 0, 1).transpose(1, 2, 0)
        feature8 = circshift(direction_os, 3, 1, 0).transpose(1, 2, 0)

        feature_list.append(feature2)
        feature_list.append(feature4)
        # feature_list.append(feature5)
        feature_list.append(feature6)
        feature_list.append(feature8)

    elif (direction_classes - 1 == 8 or direction_classes - 1 == 16):
        direction_os = direction_os.transpose(2, 0, 1)  # [2,256,256]
        feature1 = circshift(direction_os, 1, 1, 1).transpose(1, 2, 0)
        feature2 = circshift(direction_os, 1, 1, 0).transpose(1, 2, 0)
        feature3 = circshift(direction_os, 2, 1, 1).transpose(1, 2, 0)
        feature4 = circshift(direction_os, 3, 0, 1).transpose(1, 2, 0)
        feature6 = circshift(direction_os, 4, 0, 1).transpose(1, 2, 0)
        feature7 = circshift(direction_os, 3, 1, 1).transpose(1, 2, 0)
        feature8 = circshift(direction_os, 3, 1, 0).transpose(1, 2, 0)
        feature9 = circshift(direction_os, 4, 1, 1).transpose(1, 2, 0)

        feature_list.append(feature1)
        feature_list.append(feature2)
        feature_list.append(feature3)
        feature_list.append(feature4)
        # feature_list.append(feature5)
        feature_list.append(feature6)
        feature_list.append(feature7)
        feature_list.append(feature8)
        feature_list.append(feature9)

    cos_value = np.zeros((height, weight, direction_classes - 1), dtype=np.float32)
    # print('cos_value.shape = {}'.format(cos_value.shape))
    for k, feature_item in enumerate(feature_list):
        fenzi = (feature5[:, :, 0] * feature_item[:, :, 0] + feature5[:, :, 1] * feature_item[:, :, 1])
        fenmu = (np.sqrt(pow(feature5[:, :, 0], 2) + pow(feature5[:, :, 1], 2)) * np.sqrt(
            pow(feature_item[:, :, 0], 2) + pow(feature_item[:, :, 1], 2)) + 0.000001)
        cos_np = fenzi / fenmu
        cos_value[:, :, k] = cos_np

    cos_value_min = np.min(cos_value, axis=2)
    cos_sim_map = cos_value_min
    cos_sim_map[label_direction == 0] = 1

    cos_sim_map_np = (1 - np.around(cos_sim_map))
    cos_sim_map_np_max = np.max(cos_sim_map_np)
    cos_sim_map_np_min = np.min(cos_sim_map_np)
    cos_sim_map_np_normal = (cos_sim_map_np - cos_sim_map_np_min) / (cos_sim_map_np_max - cos_sim_map_np_min)

    return cos_sim_map_np_normal


if __name__ == "__main__":
    run_eval()
