#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-05 17:04:47
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-05 17:11:16
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/DCNN-MF-SP/Networks/DenseNet.py
Description: Modify here please
Init from https://github.com/Oussamayousre/automatic-classification-of-common-maternal-fetal-ultrasound-planes b784f0107fd8cd0368622c5da09a0b41d0a3eb04
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


def DenseNet(input_channels=3, img_size=(256, 256), cls_num=6, pretrained=False):
    weights = "imagenet" if pretrained else None
    DenseNet = tf.keras.applications.DenseNet121(include_top=False, weights=weights,
        input_tensor=None, input_shape=(img_size[0], img_size[1], input_channels), pooling=None)
    new_model = Sequential(name='DenseNet')
    new_model.add(DenseNet)
    new_model.add(GlobalAveragePooling2D())
    new_model.add(Dense(512))
    new_model.add(Dense(256))
    new_model.add(Dense(cls_num, activation='softmax'))
    
    new_model.summary()
    
    return new_model


if __name__ == '__main__':
    eval("DenseNet")(input_channels=3, img_size=(256, 256), cls_num=6, pretrained=True)
