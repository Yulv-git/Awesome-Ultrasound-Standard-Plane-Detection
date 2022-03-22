#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-18 10:33:38
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-22 09:46:44
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/SonoNet/inference.py
Description: PyTorch implementation of SonoNet.
Init from https://github.com/rdroste/SonoNet_PyTorch

Baumgartner et al., "Real-Time Detection and Localisation of Fetal Standard
Scan Planes in 2D Freehand Ultrasound", arXiv preprint:1612.05601 (2016)

This repository is based on https://github.com/baumgach/SonoNet-weights
which provides a theano+lasagne implementation.

inference.py:   Modified version of the original example.py file.
                This file runs classification on the examples images.
'''
import argparse
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from PIL import Image
from torch.autograd import Variable
from ipdb import set_trace

import sononet


def imcrop(image, crop_range):
    """ Crop an image to a crop range """
    return image[crop_range[0][0]:crop_range[0][1], crop_range[1][0]:crop_range[1][1], ...]


def prepare_inputs(image_path=None, crop_range=None, input_size=None):
    input_list = []
    for filename in glob.glob(image_path):
        # prepare images
        image = imread(filename)  # read
        image = imcrop(image, crop_range)  # crop
        image = np.array(Image.fromarray(image).resize(input_size, resample=Image.BICUBIC))
        image = np.mean(image, axis=2)  # convert to gray scale

        # convert to 4D tensor of type float32
        image_data = np.float32(np.reshape(image, (1, 1, image.shape[0], image.shape[1])))

        # normalise images by substracting mean and dividing by standard dev.
        mean = image_data.mean()
        std = image_data.std()
        image_data = np.array(255.0 * np.divide(image_data - mean, std), dtype=np.float32)
        # Note that the 255.0 scale factor is arbitraryit is necessary because the network was trainedlike this,
        # but the same results would have been achieved without this factor for training.

        input_list.append(image_data)

    return input_list


def main(args):
    print('\nLoading network...')
    net = sononet.SonoNet(config=args.network_name, weights=args.weights)
    net.eval()

    torch.cuda.device(args.GPU_NR)
    print('Moving to GPU: {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))
    net.cuda()

    print("\nPredictions using {}:".format(args.network_name))
    input_list = prepare_inputs(image_path=args.image_path, crop_range=args.crop_range, input_size=args.input_size)
    for image, file_name in zip(input_list, glob.glob(args.image_path)):
        x = Variable(torch.from_numpy(image).cuda())
        outputs = net(x)
        confidence, prediction = torch.max(outputs.data, 1)

        # True labels are obtained from file name.
        true_label = file_name.split('/')[-1][0:-5]
        print(" - {} (conf: {:.2f}, true label: {})".format(args.label_names[prediction[0]], confidence[0], true_label))

        if args.display_images:
            plt.imshow(np.squeeze(image), cmap='gray')
            plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--network_name', type=str, default='SN64', help=" 'SN16', 'SN32', 'SN64'")
    parse.add_argument('--display_images', type=bool, default=True, help="Whether or not to show the images during inference")
    parse.add_argument('--GPU_NR', default=0, help="Choose the device number of your GPU")
    parse.add_argument('--weights', default=True, help="(bool, 0 or string): Select weight initialization.")
    parse.add_argument('--crop_range', default=[(115, 734), (81, 874)], help="[(top, bottom), (left, right)]")
    parse.add_argument('--input_size', default=[224, 288], help="[H, W]")
    parse.add_argument('--image_path', default='./example_images/*.tiff')
    parse.add_argument('--label_names', default=['3VV',
                                                '4CH',
                                                'Abdominal',
                                                'Background',
                                                'Brain (Cb.)',
                                                'Brain (Tv.)',
                                                'Femur',
                                                'Kidneys',
                                                'Lips',
                                                'LVOT',
                                                'Profile',
                                                'RVOT',
                                                'Spine (cor.)',
                                                'Spine (sag.)'])
    args = parse.parse_args()

    main(args)
