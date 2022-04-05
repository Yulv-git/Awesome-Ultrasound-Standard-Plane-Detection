#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-03 18:28:14
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-05 17:48:14
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/DCNN-MF-SP/main.py
Description: Evaluation of Deep Convolutional Neural Networks for Automatic Classification of Common Maternal Fetal Ultrasound Planes
'''
import argparse
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from wandb.keras import WandbCallback

from utils import check_dir, recall_m, precision_m, f1_m, load_train_generator
from Networks import base_model, VGG, ResNet, DenseNet, EfficientNet, ViT


def train_val(args, train_generator, valid_generator, wandb_callback):
    # Baseline Model
    model = getattr(eval(args.model_type), args.model_name)(
        input_channels=args.input_channels, img_size=args.img_size, cls_num=args.cls_num, pretrained=args.imagenet_pretrained)
    # Plot the model Architecture.
    tf.keras.utils.plot_model(model, to_file='{}/{}.png'.format(args.save_dir, args.model_name))
    
    # Model Training
    learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', patience=2, verbose=1, factor=0.1, min_lr=0.000001)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    if args.load_model_weights:  # Load the model.
        model = tf.keras.models.load_model('{}/{}'.format(args.save_dir, args.model_name), compile=False,
                                           custom_objects={"f1_m":f1_m, "precision_m":precision_m, "recall_m":recall_m})
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    history = model.fit(train_generator, validation_data=valid_generator, epochs=args.epochs,
                        callbacks=[learning_rate_reduction, wandb_callback])
    
    # model.evaluate(valid_generator)

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
    plt.savefig('{}/{}{}_training_acc_loss.png'.format(args.save_dir, args.model_name, args.extra_string))
    
    # Save the model.
    model.save('{}/{}{}'.format(args.save_dir, args.model_name, args.extra_string), save_format='tf')


def main(args):
    args.extra_string = '_pretrained' if args.imagenet_pretrained else ''
    check_dir(args.save_dir)
    # wandb
    wandb.login(key=args.API_keys)  # Link to the platform.
    wandb.init(project=args.project, entity=args.entity)  # Initialise or track on specific project.
    wandb_callback = wandb.keras.WandbCallback(log_weights=args.log_weights)  # Pre-processing and EDA.

    # data
    train_generator, valid_generator = load_train_generator(
        imgs_info=args.imgs_info, imgs_dir=args.imgs_dir, x_col="Image_name", y_cols="Plane",
        shuffle=True, batch_size=args.batch_size, seed=1, target_w=args.img_size[0], target_h=args.img_size[1])

    train_val(args, train_generator, valid_generator, wandb_callback)


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
    parse.add_argument('--cls_num', type=int, default=6)

    # train
    parse.add_argument('--input_channels', type=int, default=3)
    parse.add_argument('--learning_rate', type=float, default=0.0001)
    parse.add_argument('--epochs', type=int, default=15)
    parse.add_argument('--save_dir', type=str, default='./output')
    parse.add_argument('--model_type', type=str, default='base_model',
                       choices=['base_model', 'VGG', 'ResNet', 'DenseNet', 'EfficientNet', 'ViT'])
    parse.add_argument('--model_name', type=str, default='base_model',
                       choices=['base_model', 'VGG19', 'ResNet50', 'DenseNet121', 'EfficientNetB6', 'ViT'])
    parse.add_argument('--imagenet_pretrained', type=bool, default=False)
    parse.add_argument('--load_model_weights', type=bool, default=False)

    args = parse.parse_args()    
    main(args)
