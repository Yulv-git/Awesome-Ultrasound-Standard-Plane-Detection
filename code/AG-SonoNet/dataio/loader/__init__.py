#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-20 18:17:37
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-02 17:50:33
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/AG-SonoNet/dataio/loader/__init__.py
Description: Modify here please
Init from https://github.com/ozan-oktay/Attention-Gated-Networks
'''
import json

from dataio.loader.ukbb_dataset import UKBBDataset
from dataio.loader.test_dataset import TestDataset
from dataio.loader.hms_dataset import HMSDataset
from dataio.loader.cmr_3D_dataset import CMR3DDataset
from dataio.loader.us_dataset import UltraSoundDataset
from dataio.loader.us_dataset import UltraSoundDataset_FPD


def get_dataset(name):
    """get_dataset
    :param name:
    """
    return {
        'ukbb_sax': CMR3DDataset,
        'acdc_sax': CMR3DDataset,
        'rvsc_sax': CMR3DDataset,
        'hms_sax':  HMSDataset,
        'test_sax': TestDataset,
        'us': UltraSoundDataset,
        'us_FPD': UltraSoundDataset_FPD
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path
    :param dataset_name:
    :param opts:
    """
    return getattr(opts, dataset_name)
