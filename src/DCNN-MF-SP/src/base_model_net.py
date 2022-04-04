#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-04 21:00:36
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-04 21:17:28
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/DCNN-MF-SP/src/base_model_net.py
Description: Modify here please
'''
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Activation


def Network(input_channels=3, img_size=(256, 256), cls_num=6):
    base_model = Sequential()
    base_model.add(Conv2D(32, (3, 3), padding="same", input_shape=(img_size[0], img_size[1], input_channels)))
    base_model.add(Activation("relu"))
    base_model.add(BatchNormalization(axis=1))
    base_model.add(MaxPooling2D(pool_size=(3, 3)))
    base_model.add(Conv2D(64, (3, 3), padding="same"))
    base_model.add(Activation("relu"))
    base_model.add(BatchNormalization(axis=1))
    base_model.add(Conv2D(64, (3, 3), padding="same"))
    base_model.add(Activation("relu"))
    base_model.add(BatchNormalization(axis=1))
    base_model.add(MaxPooling2D(pool_size=(2, 2)))
    base_model.add(Conv2D(128, (3, 3), padding="same"))
    base_model.add(Activation("relu"))
    base_model.add(BatchNormalization(axis=1))
    base_model.add(Conv2D(128, (3, 3), padding="same"))
    base_model.add(Activation("relu"))
    base_model.add(BatchNormalization(axis=1))
    base_model.add(MaxPooling2D(pool_size=(2, 2)))
    base_model.add(Flatten())
    base_model.add(Dense(1024))
    base_model.add(Activation("relu"))
    base_model.add(BatchNormalization())
    base_model.add(Dense(cls_num))
    base_model.add(Activation("softmax"))

    base_model.build((0, img_size[0], img_size[1], input_channels))

    base_model.summary()

    return base_model


if __name__ == "__main__":
    pass
