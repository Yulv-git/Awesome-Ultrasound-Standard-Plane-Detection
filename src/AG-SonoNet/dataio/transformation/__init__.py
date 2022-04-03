#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-20 18:17:37
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-02 23:15:40
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/AG-SonoNet/dataio/transformation/__init__.py
Description: Modify here please
Init from https://github.com/ozan-oktay/Attention-Gated-Networks
'''
import json

from dataio.transformation.transforms import Transformations


def get_dataset_transformation(name, opts=None):
    '''
    :param opts: augmentation parameters
    :return:
    '''
    # Build the transformation object and initialise the augmentation parameters
    trans_obj = Transformations(name)
    if opts:
        trans_obj.initialise(opts)

    # Print the input options
    trans_obj.print()

    # Returns a dictionary of transformations
    return trans_obj.get_transformation()
