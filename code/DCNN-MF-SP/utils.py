#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-04 20:37:41
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-05 17:29:15
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/DCNN-MF-SP/utils.py
Description: Modify here please
'''
import os
import pandas as pd
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator


def check_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            os.makedirs(path)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_train_generator(imgs_info='../../data/FETAL_PLANES_DB/FETAL_PLANES_DB_data.xlsx',
                         imgs_dir='../../data/FETAL_PLANES_DB/Images',
                         x_col="Image_name", y_cols="Plane", shuffle=True, batch_size=64,
                         seed=1, target_w=256, target_h=256):
    df = pd.read_excel(imgs_info, index_col=0, engine='openpyxl')  # 12400 rows × 6 columns
    df["Image_name"] = df.index  # Change the Dataframe index.  # 12400 rows × 7 columns
    df.reset_index(drop=True, inplace=True)  # Reset the Dataframe indexes.
    df.rename(columns={'Train ': 'Train'}, inplace=True)
    df.Train[df.Train == 0]  # Length: 5271, dtype: int64
    df.Image_name = imgs_dir + "/" + df.Image_name + ".png"
    df.Plane.unique()  # array(['Other', 'Maternal cervix', 'Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax'], dtype=object)
    # df['Plane'] = df['Plane'].map({'Other':0, 'Maternal cervix':1, 'Fetal abdomen':2, 'Fetal brain':3, 'Fetal femur':4, 'Fetal thorax':5})
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the Dataframe rows.

    # Split the Data.
    df_test = df[df["Train"] == 0]  # Testing data.
    df_train = df[df["Train"] == 1]  # Training Data.
    # Reset Indexes after Spliting the Data.
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Perform data augmentation here.
    image_generator = ImageDataGenerator(rescale=1/255., rotation_range=5, shear_range=0.02, zoom_range=0.02,
                                         samplewise_center=True, samplewise_std_normalization=True)
    
    # Create the image generator using flow_from_dataframe.
    train_generator=image_generator.flow_from_dataframe(
        dataframe=df_train,
        directory=None,
        x_col=x_col,
        y_col=y_cols,
        subset="training",
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        class_mode="categorical",
        target_size=(target_w, target_h))
    valid_generator=image_generator.flow_from_dataframe(
        dataframe=df_test,
        directory=None,
        x_col=x_col,
        y_col=y_cols,
        subset="training",
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        class_mode="categorical",
        target_size=(target_w, target_h))

    return train_generator, valid_generator


if __name__ == '__main__':
    pass
