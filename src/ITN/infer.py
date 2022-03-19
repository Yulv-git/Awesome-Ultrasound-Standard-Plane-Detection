#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-18 23:10:53
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-19 16:54:25
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/ITN/infer.py
Description: Modify here please
Init from https://github.com/yuanwei1989/plane-detection Author: Yuanwei Li (3 Oct 2018)
In this script, we use quaternions to represent rotation.
Reference
Standard Plane Detection in 3D Fetal Ultrasound Using an Iterative Transformation Network
https://arxiv.org/abs/1806.07486
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import inference

from utils import input_data

np.random.seed(0)

class Config(object):
    """Inference configurations."""
    # File paths
    data_dir = './data/Images'
    label_dir = './data/Landmarks'
    train_list_file = './data/list_train.txt'
    test_list_file = './data/list_test.txt'
    model_dir = './cnn_model'
    # General parameters
    plane_name = 'tv'                   # Plane name: 'tv' or 'tc'
    box_size = np.array([225, 225])     # plane size (odd number)
    input_plane = 3                     # Number of planes as input images. 1: one plane image. 3: three orthogonal plane images
    landmark_count = 16                 # Number of landmarks
    # Testing parameters
    max_test_steps = 10                 # Number of inference steps
    num_random_init = 5
    tran_weighted = True                # Whether to use classification probabilities to weight regressed translation
    rot_weighted = True                 # Whether to use classification probabilities to weight regressed rotation
    visual = True                       # Whether to save visualisation
    # Parameters for sampling training plane
    trans_frac = 0.6                    # Percentage of middle volume to sample plane centre from. (0-1)
    max_euler = [45.0 / 180.0 * np.pi,  # Maximum range to sample the three Euler angles in radians for plane orientation.
                 45.0 / 180.0 * np.pi,
                 45.0 / 180.0 * np.pi]


def main():
    config = Config()

    # Load images and ground truth planes
    data = input_data.read_data_sets(config.data_dir,
                                     config.label_dir,
                                     config.train_list_file,
                                     config.test_list_file,
                                     config.landmark_count,
                                     config.plane_name)

    print("Start inference...")
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # Load trained model
    g = tf.get_default_graph()
    saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(config.model_dir) + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
    action_prob_tran = g.get_collection('action_prob_tran')[0]  # translation classification probability
    ytr = g.get_collection('ytr')[0]                            # translation regression (displacement vector)
    action_prob_rot = g.get_collection('action_prob_rot')[0]    # rotation classification probability
    yrr_norm = g.get_collection('yrr_norm')[0]                  # rotation regression (quaternions)
    x = g.get_collection('x')[0]
    keep_prob = g.get_collection('keep_prob')[0]

    # Evaluation on test-set
    print("Evaluation on test set:")
    inference.evaluate(data.test, config, 'test',
                       sess, x, action_prob_tran, ytr, action_prob_rot, yrr_norm, keep_prob)
    # Evaluation on train-set
    print("Evaluation on train set:")
    inference.evaluate(data.train, config, 'train',
                       sess, x, action_prob_tran, ytr, action_prob_rot, yrr_norm, keep_prob)
    sess.close()


if __name__ == '__main__':
    main()
