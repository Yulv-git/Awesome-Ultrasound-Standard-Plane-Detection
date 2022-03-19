#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-19 10:33:38
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-19 17:00:40
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/ITN/utils/plane.py
Description: Functions for reading input data (image (nifti), landmarks (txt) and standard planes).
Init from https://github.com/yuanwei1989/plane-detection
'''
import numpy as np
import os
import nibabel as nib
from tensorflow.contrib.learn.python.learn.datasets import base

from utils import plane


class DataSet(object):
    def __init__(self,
                 names,
                 images,
                 landmarks,
                 trans_vecs,
                 quats,
                 pix_dim):
        self.num_examples = len(images)
        self.names = names
        self.images = images
        self.landmarks = landmarks
        self.trans_vecs = trans_vecs
        self.quats = quats
        self.pix_dim = pix_dim


def get_file_list(txt_file):
    """Get a list of filenames.
    Args:
      txt_file: Name of a txt file containing a list of filenames for the images.
    Returns:
      filenames: A list of filenames for the images.
    """
    with open(txt_file) as f:
        filenames = f.read().splitlines()
    return filenames


def extract_image(filename):
    """Extract the image into a 3D numpy array [x, y, z].
    Args:
      filename: Path and name of nifti file.
    Returns:
      data: A 3D numpy array [x, y, z]
      pix_dim: pixel spacings
    """
    img = nib.load(filename)
    data = img.get_data()
    data[np.isnan(data)] = 0
    pix_dim = np.array(img.header.get_zooms())

    return data, pix_dim


def extract_label(filename):
    """Extract the labels (landmark coordinates) into a 2D float64 numpy array.
    Args:
      filename: Path and name of txt file containing the landmarks. One row per landmark.
    Returns:
      labels: a 2D float64 numpy array. [landmark_count, 3]
    """
    with open(filename) as f:
        labels = np.empty([0, 3], dtype=np.float64)
        for line in f:
            labels = np.vstack((labels, map(float, line.split())))

    return labels


def extract_all_image_and_label(file_list,
                                data_dir,
                                label_dir,
                                landmark_count,
                                plane_name):
    """Load the input images, landmarks and the standard planes.
    Args:
      file_list: txt file containing list of filenames of images
      data_dir: Directory storing images.
      label_dir: Directory storing landmarks.
      landmark_count: Number of landmarks=16
      plane_name: 'tv' or 'tc'
    Returns:
      filenames: list of patient id names
      images: list of img_count 4D numpy arrays with dimensions=[width, height, depth, 1]. Eg. [324, 207, 279, 1]
      landmarks: landmark coordinates [img_count, landmark_count, 3]
      trans_vecs: 3D centre point of the ground truth plane. [img_count, 3]
      quats: Quaternions that rotate xy-plane to the GT plane. [img_count, 4]
      pix_dim: mm of each voxel. [img_count, 3]
    """
    filenames = get_file_list(file_list)
    file_count = len(filenames)
    images = []
    landmarks = np.zeros((file_count, landmark_count, 3))
    trans_vecs = np.zeros((file_count, 3))
    quats = np.zeros((file_count, 4))
    pix_dim = np.zeros((file_count, 3))
    for i in range(len(filenames)):
        filename = filenames[i]
        print("Loading image {}/{}: {}".format(i+1, len(filenames), filename))
        # load image
        img, pix_dim[i] = extract_image(os.path.join(data_dir, filename+'.nii.gz'))
        img_siz = np.array(img.shape)
        # load landmarks. Labels already in voxel coordinate
        landmark = extract_label(os.path.join(label_dir, filename+'_ps.txt'))
        # Compute translation and rotation of GT plane wrt reference coordinate system (origin at centre of volume)
        trans_vecs[i, :], quats[i, :], _ = plane.extract_tform(landmark, plane_name)
        trans_vecs[i, :] = trans_vecs[i, :] - (img_siz-1) / 2.0
        # Store extracted data
        images.append(np.expand_dims(img, axis=3))
        landmarks[i, :, :] = landmark

    return filenames, images, landmarks, trans_vecs, quats, pix_dim


def read_data_sets(data_dir,
                   label_dir,
                   train_list_file,
                   test_list_file,
                   landmark_count,
                   plane_name):
    """Load training and test dataset.
    Args:
      data_dir: Directory storing images.
      label_dir: Directory storing labels.
      train_list_file: txt file containing list of filenames for train images
      test_list_file: txt file containing list of filenames for test images
      landmark_count: Number of landmarks=16
      plane_name: 'tv' or 'tc'
    Returns:
      data: A collections.namedtuple containing fields ['train', 'validation', 'test']
    """
    print("Loading train images...")
    train_names, train_images, train_landmarks, train_trans_vecs, train_quats, train_pix_dim = extract_all_image_and_label(train_list_file,
                                                                                                                           data_dir,
                                                                                                                           label_dir,
                                                                                                                           landmark_count,
                                                                                                                           plane_name)
    print("Loading test images...")
    test_names, test_images, test_landmarks, test_trans_vecs, test_quats, test_pix_dim = extract_all_image_and_label(test_list_file,
                                                                                                                     data_dir,
                                                                                                                     label_dir,
                                                                                                                     landmark_count,
                                                                                                                     plane_name)
    train = DataSet(train_names, train_images, train_landmarks, train_trans_vecs, train_quats, train_pix_dim)
    test = DataSet(test_names, test_images, test_landmarks, test_trans_vecs, test_quats, test_pix_dim)

    return base.Datasets(train=train, validation=None, test=test)
