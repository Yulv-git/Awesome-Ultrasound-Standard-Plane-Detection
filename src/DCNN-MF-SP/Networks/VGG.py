#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-05 17:18:27
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-05 17:21:27
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/DCNN-MF-SP/Networks/VGG.py
Description: Modify here please
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


def VGG19(input_channels=3, img_size=(256, 256), cls_num=6, pretrained=False):
    weights = "imagenet" if pretrained else None
    VGG19 = tf.keras.applications.VGG19(include_top=False, weights=weights, input_tensor=None,
                                        input_shape=(img_size[0], img_size[1], input_channels),
                                        pooling=None, classifier_activation="softmax")
    new_model = Sequential(name='VGG19')
    new_model.add(VGG19)
    new_model.add(GlobalAveragePooling2D())
    new_model.add(Dense(512))
    new_model.add(Dense(256))
    new_model.add(Dense(cls_num, activation='softmax'))

    new_model.summary()

    return new_model


if __name__ == '__main__':
    eval("VGG19")(input_channels=3, img_size=(256, 256), cls_num=6, pretrained=True)
