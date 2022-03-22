#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-18 10:33:38
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-22 10:30:11
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/SonoNet/sononet/sononet.py
Description: PyTorch implementation of SonoNet.
Init from https://github.com/rdroste/SonoNet_PyTorch

Baumgartner et al., "Real-Time Detection and Localisation of Fetal Standard
Scan Planes in 2D Freehand Ultrasound", arXiv preprint:1612.05601 (2016)

This repository is based on https://github.com/baumgach/SonoNet-weights which
provides a theano+lasagne implementation.

PyTorch implementation of the original models.py file, plus functions to
    load and convert the lasagne weights to a PyTorch state_dict.
'''
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ipdb import set_trace


class SonoNet(nn.Module):
    """
    Args:
        config (str): Selects the architecture.
            Options are 'SN16', 'SN32' or 'SN64'
        num_labels (int, optional): Length of output vector after adaption.
            Default is 14. Ignored if features_only=True
        weights (bool, 0 or string): Select weight initialization.
            True: Load weights from default *.pth weight file.
            False: No weights are initialized.
            0: Standard random weight initialization.
            str: Pass your own weight file.
            Default is True.
        features_only (bool, optional): If True, only feature layers are initialized and the forward method returns the features.
            Default is False.
    Attributes:
        feature_channels (int): Number of feature channels.
        features (torch.nn.Sequential): Feature extraction CNN
        adaption (torch.nn.Sequential): Adaption layers for classification
    Examples::
        >>> net = sononet.SonoNet(config='SN64').eval().cuda()
        >>> outputs = net(x)

        >>> encoder = sononet.SonoNet(config='SN64', features_only=True).eval().cuda()
        >>> features = encoder(x)
    Note:
        Inputs into the forward methods must be preprocessed as shown inference.py
    """

    feature_cfg_dict = {
        'SN16': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128],
        'SN32': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256],
        'SN64': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    }

    def __init__(self, config=None, num_labels=14, weights=True, features_only=False, in_channels=1):
        super().__init__()
        self.config = config
        self.feature_cfg = self.feature_cfg_dict[config]
        self.feature_channels = self.feature_cfg[-1]
        self.weights = weights
        self.features_only = features_only
        self.features = self._make_feature_layers(self.feature_cfg, in_channels)
        if not features_only:
            self.adaption_channels = self.feature_channels // 2
            self.num_labels = num_labels
            self.adaption = self._make_adaption_layer(self.feature_channels, self.adaption_channels, self.num_labels)
        self.set_weights(weights)

    def forward(self, x):
        x = self.features(x)

        if not self.features_only:
            x = self.adaption(x)
            x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
            x = F.softmax(x, dim=1)

        return x

    @staticmethod
    def _conv_layer(in_channels, out_channels):
        layer = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                 nn.BatchNorm2d(out_channels, eps=1e-4),
                 nn.ReLU(inplace=True)]
        return nn.Sequential(*layer)

    @classmethod
    def _make_feature_layers(cls, feature_cfg, in_channels):
        layers = []
        conv_layers = []
        for v in feature_cfg:
            if v == 'M':
                conv_layers.append(nn.MaxPool2d(2))
                layers.append(nn.Sequential(*conv_layers))
                conv_layers = []
            else:
                conv_layers.append(cls._conv_layer(in_channels, v))
                in_channels = v
        layers.append(nn.Sequential(*conv_layers))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_adaption_layer(feature_channels, adaption_channels, num_labels):
        return nn.Sequential(
            nn.Conv2d(feature_channels, adaption_channels, 1, bias=False),
            nn.BatchNorm2d(adaption_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(adaption_channels, num_labels, 1, bias=False),
            nn.BatchNorm2d(num_labels),
        )

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    @staticmethod
    def process_lasagne_weights(weights):
        order = [0, 2, 1, 3, 4]
        weights = [weights[5 * (idx // 5) + order[idx % 5]] for idx in range(len(weights))]
        weights[4::5] = [np.power(w, -2) - 1e-4 for w in weights[4::5]]
        return weights

    @classmethod
    def load_lasagne_weights(cls, filename, state):
        with np.load(filename) as f:
            weight_data = [f['arr_%d' % i] for i in range(len(f.files))]
        weight_data = cls.process_lasagne_weights(weight_data)
        offset = 0
        for idx, layer in enumerate(state):
            if 'num_batches_tracked' in layer:
                offset -= 1
                continue
            # assert tuple(state[layer].shape) == weight_data[idx].shape
            if not tuple(state[layer].shape) == weight_data[idx + offset].shape:
                pass
            state[layer] = torch.from_numpy(weight_data[idx + offset].copy())

    @staticmethod
    def save_state(state, filename):
        if (not os.path.isfile(filename) or input('Overwrite state file?\nHit [y] to continue: ') == 'y'):
            torch.save(state, filename)

    def load_weights(self, weights):
        _, extension = os.path.splitext(weights)
        if extension == '.npz':
            state = self.state_dict()
            state = self.load_lasagne_weights(weights, state)
            self.save_state(state, os.path.join(os.path.dirname(__file__), 'SonoNet{}.pth'.format(self.config[2:])))
        elif extension == '.pth':
            state = torch.load(weights)
        else:
            raise ValueError('Unknown weight file extension {}'.format(extension))
        # Check if input channels match
        # for key in self.state_dict():
        #     size = self.state_dict()[key].size()
        #     if state[key].size() != size:
        #         expand = state[key].expand(size)
        #         state[key] = expand * state[key].norm() / expand.norm()
        self.load_state_dict(state, strict=True)

    def set_weights(self, weights):
        if weights is not None:
            if weights:
                if not isinstance(weights, str):
                    weights = os.path.join(os.path.dirname(__file__), 'SonoNet{}.pth'.format(self.config[2:]))
                self.load_weights(weights)
            else:
                self.apply(self._initialize_weights)


if __name__ == '__main__':
    label_names=['3VV', '4CH', 'Abdominal', 'Background', 'Brain (Cb.)', 'Brain (Tv.)', 'Femur',
                'Kidneys', 'Lips', 'LVOT', 'Profile', 'RVOT', 'Spine (cor.)', 'Spine (sag.)']
    image = np.random.randint(0, 255, (1,1,288,224), dtype=np.int32)
    image = np.float32(image)
    image = np.array(255.0 * np.divide(image - 50.57153, 51.57832), dtype=np.float32)
    x = Variable(torch.from_numpy(image).cuda())

    net = SonoNet(config='SN64').eval().cuda()
    outputs = net(x)
    confidence, prediction = torch.max(outputs.data, 1)
    print(" - {} (conf: {:.2f})".format(label_names[prediction[0]], confidence[0]))
