#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-04 21:00:36
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-08 22:10:15
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/DCNN-MF-SP/Networks/base_model.py
Description: Init from https://github.com/Oussamayousre/automatic-classification-of-common-maternal-fetal-ultrasound-planes b784f0107fd8cd0368622c5da09a0b41d0a3eb04
'''
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation


def base_model(model_name='base_model', input_channels=3, img_size=(256, 256), cls_num=6, pretrained=False):
    model = Sequential(name=model_name)
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(img_size[0], img_size[1], input_channels)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dense(cls_num))
    model.add(Activation("softmax"))

    model.build((0, img_size[0], img_size[1], input_channels))

    model.summary()

    return model


if __name__ == "__main__":
    base_model(model_name='base_model', input_channels=3, img_size=(256, 256), cls_num=6)
