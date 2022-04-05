#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-05 16:00:21
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-05 22:57:26
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/DCNN-MF-SP/Networks/EfficientNet.py
Description: Modify here please
Init from https://github.com/Oussamayousre/automatic-classification-of-common-maternal-fetal-ultrasound-planes b784f0107fd8cd0368622c5da09a0b41d0a3eb04
'''
import tensorflow as tf
from tensorflow.keras import layers, applications


def EfficientNet(model_name='EfficientNetB6', input_channels=3, img_size=(256, 256), cls_num=6, pretrained=False):
    inputs = layers.Input(shape=(img_size[0], img_size[1], input_channels))
    weights = "imagenet" if pretrained else None
    outputs = getattr(applications, model_name)(include_top=False, weights=weights)(inputs)

    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(512, activation='relu')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.Dense(256, activation='relu')(outputs)
    outputs = layers.Dense(cls_num, activation='softmax')(outputs)

    EfficientNet = tf.keras.Model(inputs, outputs)

    EfficientNet.summary()
    
    return EfficientNet


if __name__ == '__main__':
    EfficientNet(model_name='EfficientNetB7', input_channels=3, img_size=(256, 256), cls_num=6, pretrained=True)
