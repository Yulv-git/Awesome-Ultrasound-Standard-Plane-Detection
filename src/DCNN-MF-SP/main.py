#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-03 18:28:14
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-03 23:05:17
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
from sklearn.model_selection import train_test_split
from keras_preprocessing.image.dataframe_iterator import DataFrameIterator
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from keras.utils import data_utils
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D,GlobalAveragePooling2D
from glob import glob
from keras import backend as K
from tensorflow.keras.models import load_model
from wandb.keras import WandbCallback


class ImageFrameGenerator(tf.keras.utils.Sequence):  # Generates data for Keras.
    def __init__(self, train_ID, batch_size=1, image_list_path=image_train_array, label_list=label_train_array,
                 image_batch=32, dataset=df_test, dim=(256,256), shuffle=True):
        self.batch_size = batch_size
        self.image_list_path = image_list_path
        self.label_list = label_list
        self.image_batch = image_batch
        self.train_ID = train_ID   
        self.dataset = dataset
        self.shuffle = shuffle
        self.dim = dim
        self.on_epoch_end()

    def __len__(self):  # Denotes the number of batches per epoch.
        return int(np.floor(len(self.train_ID)))

    def __getitem__(self, index):  # Generate one batch of data.
        # Generate indexes of the batch.
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs.
        list_IDs_temp = [k for k in indexes]
        # list_IDs_temp = [20]
        # Generate data.
        data, labels = self.__data_generation(list_IDs_temp)

        return data/255, labels

    def on_epoch_end(self):  # Updates indexes after each epoch.
        self.indexes = np.arange(len(self.train_ID))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):  # Generates data containing batch_size samples.  # X: (n_samples, *dim, n_channels)
        data = np.array([]).reshape(0,256,256,3)
        labels = np.array([])
        first_index = list_IDs_temp[0] * self.image_batch
        for i in np.arange(32):
            # img = tf.keras.preprocessing.image.array_to_img(data)
            image_name = f"./Images/{self.image_list_path[first_index+i]}.png" 
            image = cv2.imread(image_name)
            image = cv2.resize(image,(256,256))
            image = image.reshape((1,256,256,3))
            data = np.vstack([data,image])
            labels = np.append(labels,self.label_list[first_index+i])

        return data, labels


def main(args):
    # wandb
    wandb.login(key=args.API_keys)  # Link to the platform.
    wandb.init(project=args.project, entity=args.entity)  # Initialise or track on specific project.
    wandb_callback = wandb.keras.WandbCallback(log_weights=args.log_weights)  # Pre-processing and EDA.

    # data
    list_images = os.listdir(args.imgs_dir)  # len(list_images)  # 12400
    df = pd.read_excel(args.imgs_info, index_col=0, engine='openpyxl')  # 12400 rows × 6 columns
    df["Image_name"] = df.index  # Change the Dataframe index.  # 12400 rows × 7 columns
    df.reset_index(drop=True, inplace=True)  # Reset the Dataframe indexes.
    df.rename(columns={'Train ': 'Train'}, inplace=True)
    df.Train[df.Train == 0]  # Length: 5271, dtype: int64
    df.Image_name = "{}/{}.png".format(args.imgs_dir, df.Image_name)
    df.Plane.unique()  # array(['Other', 'Maternal cervix', 'Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax'], dtype=object)
    # df['Plane'] = df['Plane'].map({'Other':0, 'Maternal cervix':1, 'Fetal abdomen':2, 'Fetal brain':3, 'Fetal femur':4, 'Fetal thorax':5})
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the Dataframe rows.

    # Split the Data.
    df_test = df[df["Train"] == 0]  # Testing data.
    df_train = df[df["Train"] == 1]  # Training Data.
    # Reset Indexes after Spliting the Data.
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    image_train_array = df_train.Image_name.to_numpy()  # 7129
    label_train_array = df_train.Plane.to_numpy()
    image_test_array = df_test.Image_name.to_numpy()
    label_test_array = df_test.Plane.to_numpy()

    # Image Data generation using Sequence.
    train_params = {'dim':(256, 256), 'batch_size':1, 'shuffle':False, 'dataset':df_train,
                    'image_list_path':image_train_array, 'label_list':label_train_array}
    test_params = {'dim':(256,256), 'batch_size':1, 'shuffle':False, 'dataset':df_test,
                   'image_list_path':image_test_array, 'label_list':label_test_array}
    training_generator = ImageFrameGenerator(np.arange(df_train.Train.size//32 - 1), **train_params)
    testing_generator = ImageFrameGenerator(np.arange(df_test.Train.size//32 - 1), **test_params)
    datagen=ImageDataGenerator(rescale=1./255., validation_split=0.25)


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
    
    args = parse.parse_args()
    main(args)
