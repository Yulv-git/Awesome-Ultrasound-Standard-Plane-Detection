#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-05 23:08:55
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-05 23:35:04
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/DCNN-MF-SP/Networks/DCNN.py
Description: Modify here please
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import applications


def DCNN(model_name='ResNet50', input_channels=3, img_size=(256, 256), cls_num=6, pretrained=False):
    weights = "imagenet" if pretrained else None
    BackBone = getattr(applications, model_name)(
        include_top=False, weights=weights, input_shape=(img_size[0], img_size[1], input_channels))

    model = Sequential(name=model_name)
    model.add(BackBone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(cls_num, activation='softmax'))

    model.summary()

    return model


if __name__ == '__main__':
    DCNN(model_name='ResNet50', input_channels=3, img_size=(256, 256), cls_num=6, pretrained=True)
