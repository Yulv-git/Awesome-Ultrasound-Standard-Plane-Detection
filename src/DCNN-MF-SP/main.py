#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-03 18:28:14
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-04 21:40:06
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/DCNN-MF-SP/main.py
Description: Evaluation of Deep Convolutional Neural Networks for Automatic Classification of Common Maternal Fetal Ultrasound Planes
Init from https://github.com/Oussamayousre/automatic-classification-of-common-maternal-fetal-ultrasound-planes b784f0107fd8cd0368622c5da09a0b41d0a3eb04
'''
import argparse
import os
import sys 
import json
import random
import collections
import time
import re
import cv2
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom as dicom
import tensorflow as tf
import wandb
from ipdb import set_trace
from sklearn.model_selection import train_test_split
from keras_preprocessing.image.dataframe_iterator import DataFrameIterator
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from keras.utils import data_utils
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.core import Activation
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, GlobalAveragePooling2D
from glob import glob
from keras import backend as K
from tensorflow.keras.models import load_model
from wandb.keras import WandbCallback

from src.utils import check_dir, recall_m, precision_m, f1_m, load_train_generator
from src.base_model_net import Network as base_network


def main(args):
    # wandb
    wandb.login(key=args.API_keys)  # Link to the platform.
    wandb.init(project=args.project, entity=args.entity)  # Initialise or track on specific project.
    wandb_callback = wandb.keras.WandbCallback(log_weights=args.log_weights)  # Pre-processing and EDA.

    # data
    train_generator, valid_generator = load_train_generator(
        imgs_info=args.imgs_info, imgs_dir=args.imgs_dir, x_col="Image_name", y_cols="Plane",
        shuffle=True, batch_size=args.batch_size, seed=1, target_w=args.img_size[0], target_h=args.img_size[1])

    # Baseline Model
    base_model = base_network(input_channels=args.input_channels, img_size=args.img_size, cls_num=args.cls_num)
    # Plot the model Architecture.
    tf.keras.utils.plot_model(base_model, to_file='{}/base_model.png'.format(args.save_dir))
    
    # Model Training
    learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', patience=2, verbose=1, factor=0.1, min_lr=0.000001)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    base_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    history = base_model.fit(train_generator, validation_data=valid_generator, epochs=args.epochs, callbacks=[learning_rate_reduction, wandb_callback])

    # Plot loss during training.
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # Plot accuracy during training.
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.savefig('{}/base_model_training_acc_loss.png'.format(args.save_dir))
    
    # Save the model.
    base_model.save('{}/base_model'.format(args.save_dir), save_format='tf')
    # Load the model.
    model = tf.keras.models.load_model('{}/base_model'.format(args.save_dir), compile=False)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='')
    # wandb parameters
    parse.add_argument('--API_keys', type=str, default='f425e6f003db01e57ff6f92422528397116e3f7b', help="Your wandb API keys")
    parse.add_argument('--project', type=str, default='DCNN-MF-SP')
    parse.add_argument('--entity', type=str, default='yulv')
    parse.add_argument('--log_weights', type=bool, default=True)
    
    # data
    parse.add_argument('--imgs_dir', type=str, default='../../data/FETAL_PLANES_DB/Images')
    parse.add_argument('--imgs_info', type=str, default='../../data/FETAL_PLANES_DB/FETAL_PLANES_DB_data.xlsx')
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--img_size', default=(256, 256), help='(target_w, target_h)')
    parse.add_argument('--cls_num', default=6)

    #
    parse.add_argument('--input_channels', default=3)
    parse.add_argument('--learning_rate', default=0.0001)
    parse.add_argument('--epochs', default=15)
    parse.add_argument('--save_dir', default='./output')

    args = parse.parse_args()
    check_dir(args.save_dir)
    
    main(args)
