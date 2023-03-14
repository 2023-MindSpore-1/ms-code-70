# -*- coding: utf-8 -*-
"""
@author: huangxs
@License: (C)Copyright 2021, huangxs
@CreateTime: 2021/11/16 16:59:35
@Filename: cdnet
service api views
"""
import os

import mindspore.nn as nn
import mindspore.ops.functional as F
import mindspore.ops.operations as P

from mindspore import dtype as mstype
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore.common.initializer import One, Normal

from .models.vgg16.config import get_config_static
from .models.vgg16.vgg import vgg16

from mindspore import load_checkpoint, load_param_into_net


def get_backbone(name, pretrained=False):
    if name == 'vgg16_bn':
        _config_imagenet2012 = get_config_static(config_path="imagenet2012_config_bn.yaml")
        _vgg16_imagenet2012 = vgg16(num_classes=_config_imagenet2012.num_classes, args=_config_imagenet2012)
        if pretrained:
            _path = "vgg16_bn.ckpt"
            if not _path.startswith('/'):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                _path = os.path.join(current_dir, 'models', 'vgg16', _path)
            param_dict = load_checkpoint(_path)
            load_param_into_net(_vgg16_imagenet2012, param_dict)
        backbone = _vgg16_imagenet2012.layers

        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
        return backbone, feature_names, backbone_output


def conv_bn_relu(in_channel, out_channel, use_bn=True, kernel_size=3, stride=1, pad_mode="same", activation='relu'):
    output = []
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode=pad_mode))
    nn.Conv2dTranspose(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                       pad_mode='pad', padding=1, has_bias=(not use_bn))
    if use_bn:
        output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)


class UnetConv2d(nn.Cell):
    """
    Convolution block in Unet, usually double conv.
    """

    def __init__(self, in_channel, out_channel, use_bn=True, num_layer=2, kernel_size=3, stride=1, padding='same'):
        super(UnetConv2d, self).__init__()
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channel
        self.out_channel = out_channel

        convs = []
        for _ in range(num_layer):
            convs.append(conv_bn_relu(in_channel, out_channel, use_bn, kernel_size, stride, padding, "relu"))
            in_channel = out_channel

        self.convs = nn.SequentialCell(convs)

    def construct(self, inputs):
        x = self.convs(inputs)
        return x


class UnetUp(nn.Cell):
    """
    Upsampling high_feature with factor=2 and concat with low feature
    """

    def __init__(self, in_channel, out_channel, use_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2d(in_channel + (n_concat - 2) * out_channel, out_channel, False)
        self.concat = P.Concat(axis=1)
        self.use_deconv = use_deconv
        if use_deconv:
            self.up_conv = nn.Conv2dTranspose(in_channel, out_channel, kernel_size=2, stride=2, pad_mode="same")
        else:
            self.up_conv = nn.Conv2d(in_channel, out_channel, 1)

    def construct(self, high_feature, *low_feature):
        if self.use_deconv:
            output = self.up_conv(high_feature)
        else:
            _, _, h, w = F.shape(high_feature)
            output = P.ResizeBilinear((h * 2, w * 2))(high_feature)
            output = self.up_conv(output)
        for feature in low_feature:
            output = self.concat((output, feature))
        return self.conv(output)


class UpsampleBlock(nn.Cell):
    # 已完成mindspore的切换
    # TODO: separate parametric and non-parametric classes?
    # TODO: skip connection concatenated OR added

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in / 2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.Conv2dTranspose(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4), stride=2,
                                         pad_mode='pad', padding=1, has_bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3), stride=1, pad_mode='pad',
                                   padding=1, has_bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU()

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3), stride=1, pad_mode='pad',
                               padding=1, has_bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    # def forward(self, x, skip_connection=None): #original code
    def construct(self, x, skip_connection=1):  # hhl revised
        x = self.up(x)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            diffY = skip_connection.shape[2] - x.shape[2]
            diffX = skip_connection.shape[3] - x.shape[3]
            # print('befroe', x.shape, skip_connection.shape)
            x = ops.Pad(((0, 0), (0, 0), (diffX // 2, diffX - diffX // 2), (diffY // 2, diffY - diffY // 2)))(x)
            # print('after', x.shape, ((0,0), (0,0), (diffX // 2, diffX - diffX // 2), (diffY // 2, diffY - diffY // 2)))

            x = ops.Concat(axis=1)((x, skip_connection))

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=False)


class ResidualUnit(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(ResidualUnit, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def construct(self, x):
        residual = self.conv_1x1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out


class revAttention(nn.Cell):  # sSE
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, has_bias=False)
        self.norm = nn.Sigmoid()

    def construct(self, U, V):
        q = self.Conv1x1(V)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * (1 + q)


class CDNet(nn.Cell):
    """
    Simple UNet with skip connection
    """

    def __init__(self,
                 backbone_name='vgg16_bn',
                 encoder_freeze=False,
                 classes=21,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True
                 ):
        super(CDNet, self).__init__()

        self.backbone_name = backbone_name
        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(self.backbone_name)

        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.CellList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)

        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            # print('upsample_blocks[{}] in: {}   out: {}'.format(i, filters_in, filters_out))
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks - i - 1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm))

        self.replaced_conv1 = False  # for accommodating  inputs with different number of channels later

        self.mask_feature = ResidualUnit(decoder_filters[-1], 64)
        self.direction_feature = ResidualUnit(64, 64)
        self.point_feature = ResidualUnit(64, 64)
        self.point_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.directionAtt = revAttention(1)
        self.direction_conv = nn.Conv2d(64, 9, kernel_size=1)
        self.maskAtt = revAttention(9)
        self.mask_conv = nn.Conv2d(64, 3, kernel_size=1)

    def freeze_encoder(self):
        print('encode参数冻结未改写')

    def construct(self, _input):
        x, features = self.forward_backbone(_input)

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x_F1 = self.mask_feature(x)
        x_F2 = self.direction_feature(x_F1)
        x_F3 = self.point_feature(x_F2)
        # out[1] = x_point
        x_point = self.point_conv(x_F3)
        x_F2_direction = self.directionAtt(x_F2, x_point)
        # out[2] = x_direction
        x_direction = self.direction_conv(x_F2_direction)
        x_F1_mask = self.maskAtt(x_F1, x_direction)
        x_final_mask = self.mask_conv(x_F1_mask)

        return x_final_mask, x_point, x_direction

    def forward_backbone(self, x):
        """ Forward propagation in backbone encoder network.  """
        features = {None: None} if None in self.shortcut_features else dict()

        for name in self.backbone.name_cells():
            child = self.backbone.name_cells()[name]

            x = child(x)

            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """
        x = ops.Zeros()((1, 3, 224, 224), mstype.float32)

        has_fullres_features = 'vgg' in self.backbone_name or self.backbone_name == 'unet_encoder'
        # only VGG has features at full resolution
        channels = [] if has_fullres_features else [0]
        out_channels = ''

        for name in self.backbone.name_cells():
            child = self.backbone.name_cells()[name]
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels


if __name__ == "__main__":
    _cdnet = CDNet(backbone_name='vgg16_bn', pretrained=True, encoder_freeze=False, classes=3)

    _input = Tensor(shape=(8, 3, 256, 256), dtype=mstype.float32, init=Normal())
    _output = _cdnet(_input)
    x_final_mask, x_point, x_direction = _output
    print(x_final_mask.shape, x_point.shape, x_direction.shape)
