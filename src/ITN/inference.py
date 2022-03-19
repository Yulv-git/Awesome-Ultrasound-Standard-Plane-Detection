#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-19 10:33:38
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-19 16:53:45
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/ITN/inference.py
Description: Functions for inference.
Init from https://github.com/yuanwei1989/plane-detection
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import time
from skimage.measure import compare_ssim

import srmg.core.RiemannianLeft as RL
import srmg.common.util as srmg_util
from utils import geometry, plane, save, visual


def evaluate(data, config, suffix,
             sess, x, action_prob_tran, ytr, action_prob_rot, yrr_norm, keep_prob):
    """Using CNN to find the plane iteratively and performs evaluation

        Args:
            data: 3D volume
            config: inference parameters
            suffix: 'test' or 'train'
            sess, x, action_prob_tran, ytr, action_prob_rot, yrr_norm, keep_prob: Tensorflow CNN prediction nodes

    """
    names = data.names
    img_count = len(data.images)
    trans_vecs_final = np.zeros((img_count, 3))
    quats_final = np.zeros((img_count, 4))
    trans_vecs_gt = data.trans_vecs
    quats_gt = data.quats
    time_elapsed = np.zeros(img_count)
    box_size = config.box_size
    box_size_large = np.array([225, 225])       # larger plane size for evaluation and visualisation

    for i in xrange(img_count):
        image = np.squeeze(data.images[i])

        # Predict plane
        start_time = time.time()
        slices, meshes, matrices = predict_plane(image, config,
                                                 sess, x, action_prob_tran, ytr, action_prob_rot, yrr_norm, keep_prob)
        trans_vecs_final[i], quats_final[i] = calc_mean(matrices[:, -1, :, :], mode='r')
        end_time = time.time()
        time_elapsed[i] = end_time - start_time
        print("Image {}/{}: {}, time = {:.10f}s".format(i + 1, img_count, names[i], time_elapsed[i]))

        # Visualisation
        if config.visual:
            img_siz = np.array(image.shape)

            # Extract GT and predicted plane images
            slice_gt, mesh_gt = plane.extract_plane_from_pose(image, trans_vecs_gt[i], quats_gt[i], box_size, 1)
            slice_final, mesh_final = plane.extract_plane_from_pose(image, trans_vecs_final[i], quats_final[i], box_size, 1)

            # Plot visualisations
            visual.plot_planes('./results/plane_visual', suffix, names[i], img_siz,
                               trans_vecs_final[i], quats_final[i], mesh_final,
                               trans_vecs_gt[i], quats_gt[i], mesh_gt)
            visual.plot_images('./results/image_visual', suffix, names[i], slice_final, slice_gt)
            visual.plot_planes_movie('./results/plane_movie', suffix, names[i], img_siz, config.max_test_steps,
                                     meshes, matrices, trans_vecs_gt[i], quats_gt[i], mesh_gt)
            visual.plot_images_movie('./results/image_movie', suffix, names[i], config.max_test_steps, slices, slice_gt)

            # Save GT and predicted plane images as NIfTI files
            save.save_planes_nifti('./results/planes_nifti', suffix, names[i], slice_final, slice_gt)

    # Time
    print("Mean running time = {:.10f}s\n".format(np.mean(time_elapsed)))

    # Evaluation of predicted planes
    dist_err, angle_err, mse, psnr, ssim = compute_err(trans_vecs_final, quats_final,
                                                       trans_vecs_gt, quats_gt,
                                                       data.images, data.pix_dim, box_size_large)

    # Save evaluation
    save.save_err('./results/evaluation', suffix, names, dist_err, angle_err, mse, psnr, ssim)

    # Save plane as 4x4 transformation matrix
    save.save_planes_tform('./results/planes_tform', suffix, names, trans_vecs_final, quats_final)


def predict_plane(image, config,
                  sess, x, action_prob_tran, ytr, action_prob_rot, yrr_norm, keep_prob):
    """Using CNN to find the plane iteratively

        Args:
            image: 3D volume
            config: inference parameters
            sess, x, action_prob_tran, ytr, action_prob_rot, yrr_norm, keep_prob: Tensorflow CNN prediction nodes

        Returns:
            slices: 2D plane images at each iteration [init_count, max_test_steps+1, mesh_ind, mesh_size[0], mesh_size[1]]
            meshes: mesh coordinates of the plane found at each iteration. [init_count, max_test_steps+1, mesh_ind, mesh_siz[0], mesh_siz[1], 3]. volume corner as origin
            matrices: the 4x4 transformation matrices at each iteration. [init_count, max_test_steps+1, 4, 4]. volume centre as origin

    """
    mesh_siz = config.box_size
    init_count = config.num_random_init
    max_test_steps = config.max_test_steps
    img_siz = np.array(image.shape)
    slices = np.zeros((init_count, max_test_steps+1, config.input_plane, mesh_siz[0], mesh_siz[1]))
    meshes = np.zeros((init_count, max_test_steps+1, config.input_plane, mesh_siz[0], mesh_siz[1], 3))
    matrices = np.zeros((init_count, max_test_steps+1, 4, 4))
    ytr_vals = np.zeros((init_count, max_test_steps, 3))
    ytc_vals = np.zeros((init_count, max_test_steps, 6))
    yrr_vals = np.zeros((init_count, max_test_steps, 4))
    yrc_vals = np.zeros((init_count, max_test_steps, 6))

    # Initialise pose: randomly sample translation (plane centre) and quaternions
    eu_angles_init = geometry.sample_euler_angles_fix_range(init_count, config.max_euler[0], config.max_euler[1], config.max_euler[2])
    trans_vec_init = (np.random.rand(init_count, 3) * (img_siz * config.trans_frac) + img_siz * (1 - config.trans_frac) / 2.0) - ((img_siz - 1) / 2.0)
    for i in xrange(init_count):
        matrices[i, 0, :, :] = geometry.euler_matrix(eu_angles_init[i, 0], eu_angles_init[i, 1], eu_angles_init[i, 2], axes='rxyz')
        matrices[i, 0, :3, 3] = trans_vec_init[i]

    # Initialise mesh coordinates
    if config.input_plane == 1:
        mesh_init = np.expand_dims(plane.init_mesh_by_plane(mesh_siz, 'z'), axis=0)  # mesh_init=[1, 4, num_mesh_pts]
    elif config.input_plane == 3:
        mesh_init = plane.init_mesh_ortho(mesh_siz)   # mesh_init=[3, 4, num_mesh_pts]
    mesh_current = np.matmul(np.expand_dims(matrices[:, 0, :, :], axis=1),
                             np.expand_dims(mesh_init, axis=0))         # mesh_current=[init_count, mesh_ind, 4, num_mesh_pts]

    #  Extract initial plane image
    slices[:, 0, :, :, :], meshes[:, 0, :, :, :, :] = plane.extract_plane_from_mesh_ortho_batch(image, mesh_current, mesh_siz, 1)

    # Iterative prediction
    for i in xrange(max_test_steps):
        # CNN predictions
        ytc_vals[:, i, :], ytr_vals[:, i, :], yrc_vals[:, i, :], yrr_vals[:, i, :] = sess.run([action_prob_tran, ytr, action_prob_rot, yrr_norm],
                                                                                              feed_dict={x: slices[:, i, :, :, :].transpose((0, 2, 3, 1)), keep_prob: 1.0})

        # Form transformation matrix, mat_diff
        mat_diff = predict_mat_diff(ytc=ytc_vals[:, i, :], ytr=ytr_vals[:, i, :],
                                    yrc=yrc_vals[:, i, :], yrr=yrr_vals[:, i, :],
                                    weight_tran=config.tran_weighted, weight_rot=config.rot_weighted)

        # Pose composition. Add on predicted pose
        matrices[:, i+1, :, :] = np.matmul(matrices[:, i, :, :], mat_diff)

        # Pose point composition. Map identity plane to current plane.
        mesh_current = np.matmul(np.expand_dims(matrices[:, i+1, :, :], axis=1),
                                 np.expand_dims(mesh_init, axis=0))     # [init_count, mesh_ind, 4, num_mesh_pts]

        # Extract plane image
        slices[:, i+1, :, :, :], meshes[:, i+1, :, :, :, :] = plane.extract_plane_from_mesh_ortho_batch(image, mesh_current, mesh_siz, 1)

    return slices, meshes, matrices


def predict_mat_diff(ytc=None, ytr=None, yrc=None, yrr=None, weight_tran=False, weight_rot=False):
    """Form predicted transformation matrix from translation vector and quaternions.
       Use classification probabilities as weighting if weight_tran, weight_rot set to True

    Args:
      ytc: predicted translation probabilities [num_examples, 6]
      ytr: predicted translation vector [num_examples, 3].
      yrc: predicted rotation probabilities [num_examples, 6]
      yrr: predicted rotation quaternions [num_examples, 4].
      weight_tran: False: Regression only.
                   True: Soft classification. Multiply classification probabilities with regressed distances.
      weight_rot:  False: Regression only.
                   True: Soft classification. Multiply classification probabilities with regressed rotation.

    Returns:
      mat_diff: Predicted transformation matrix [num_examples, 4, 4]

    """
    num_examples = yrr.shape[0]
    mat_diff = np.zeros((num_examples, 4, 4))

    # Rotation prediction
    if (not weight_rot) or (yrc is None):
        # Regression only.
        for j in xrange(num_examples):
            mat_diff[j] = geometry.quaternion_matrix(yrr[j])
    else:
        # Multiply classification probabilities with regressed rotation.
        yrc_max = np.amax(yrc, axis=1)  # yrc_max=[num_examples]
        rot_axis = (np.argmax(yrc, axis=1) / 2).astype(int)  # rot_axis=[num_examples]

        # Convert predicted quaternions to euler angles using the most probable convention
        # Then convert to rotation matrix
        for j in xrange(num_examples):
            if rot_axis[j] == 0:
                euler = np.array(geometry.euler_from_quaternion(yrr[j, :], axes='sxyz'))
                euler[1:3] = 0
                euler[0] = euler[0] * yrc_max[j]
                mat_diff[j] = geometry.euler_matrix(euler[0], euler[1], euler[2], axes='sxyz')
            elif rot_axis[j] == 1:
                euler = np.array(geometry.euler_from_quaternion(yrr[j, :], axes='syxz'))
                euler[1:3] = 0
                euler[0] = euler[0] * yrc_max[j]
                mat_diff[j] = geometry.euler_matrix(euler[0], euler[1], euler[2], axes='syxz')
            elif rot_axis[j] == 2:
                euler = np.array(geometry.euler_from_quaternion(yrr[j, :], axes='szxy'))
                euler[1:3] = 0
                euler[0] = euler[0] * yrc_max[j]
                mat_diff[j] = geometry.euler_matrix(euler[0], euler[1], euler[2], axes='szxy')

    # Translation prediction
    if (not weight_tran) or (ytc is None):
        # Regression only.
        mat_diff[:, :3, 3] = ytr
    else:
        # Soft classification. Multiply classification probabilities with regressed distances.
        mat_diff[:, :3, 3] = ytr * np.amax(np.reshape(ytc, (ytc.shape[0], 3, 2)), axis=2)

    return mat_diff


def calc_mean(matrices, mode):
    """Compute the mean for a set of SE3 objects.

    Args:
      matrices: a set of 4x4 transformation matrices to be averaged [init_count, 4, 4]
      mode: 'r': Riemmanian mean
            'e': Euclidean mean

    Returns:
      trans_vecs_final: mean translation vector [3]
      quats_final: mean quaternion [4]

    """
    if mode == 'r':
        # Prepare input data (set of poses to be averaged)
        init_count = matrices.shape[0]
        poses = np.zeros((init_count, 6))  # [init_count, 6]. Rotation vector followed by translation vector.
        poses[:, 3:] = matrices[:, :3, 3]  # Translation vector
        for j in xrange(init_count):  # Rotation vector
            poses[j, :3] = srmg_util.rotVect(matrices[j, :3, :3])

        # Compute Riemannian mean
        a = 1
        w = np.ones([init_count]) / init_count  # w: N vector of SE3 point weightage
        pose_final = RL.frechetL(a, poses, w)
        trans_vecs_final = pose_final[3:]
        quats_final = geometry.quaternion_about_axis(geometry.vector_norm(pose_final[:3]),
                                                    geometry.unit_vector(pose_final[:3]))

    elif mode == 'e':
        init_count = matrices.shape[0]
        trans_vecs = matrices[:, :3, 3]  # Translation vector
        quats = np.zeros((init_count, 4))
        for j in xrange(init_count):  # Rotation vector
            quats[j, :] = geometry.quaternion_from_matrix(matrices[j], isprecise=True)
        trans_vecs_final = np.mean(trans_vecs, axis=0)
        quats_final = np.mean(quats, axis=0)

    else:
        raise ValueError('Invalid mode for computing mean.')

    return trans_vecs_final, quats_final


def compute_err(trans_vecs, quats, trans_vecs_gt, quats_gt, images, pix_dim, box_size):
    """Evaluation between predicted and GT planes.

    Args:
      trans_vecs: Predicted translation vector [img_count, 3]
      quats: Predicted quaternions [img_count, 4]
      trans_vecs_gt: GT translation vector [img_count, 3]
      quats_gt: GT quaternions [img_count, 4]
      images: a list of 3D images with dimension [width, height, depth, 1]
      pix_dim: Pixel spacing. [img_count, 3]
      box_size: Size of plane image. [2]

    Returns:
      dist_err: distance error in mm. [img_count, num_landmarks]
      angle_err: angle error in degree. [img_count, num_landmarks]
      mse: mean squared error between images
      psnr: psnr between images
      ssim: ssim between images

    """
    img_count = len(images)

    # Distance between plane centres
    dist_err = np.sqrt(np.sum(((trans_vecs - trans_vecs_gt) * pix_dim) ** 2, axis=1))

    # Angle between normal vectors of planes
    np.arccos((quats * quats_gt).sum(axis=1))
    angle_err = 2 * np.arccos((quats * quats_gt).sum(axis=1)) / np.pi * 180.0

    # Image MSE, PSNR, SSIM
    mse = np.zeros(img_count)
    psnr = np.zeros(img_count)
    ssim = np.zeros(img_count)
    for i in xrange(img_count):
        slice_gt, mesh_gt = plane.extract_plane_from_pose(images[i][..., 0], trans_vecs_gt[i], quats_gt[i], box_size, 1)
        slice_final, mesh_final = plane.extract_plane_from_pose(images[i][..., 0], trans_vecs[i], quats[i], box_size, 1)
        mse[i] = np.mean((slice_final - slice_gt) ** 2)
        psnr[i] = 20 * np.log10(1.0 / np.sqrt(mse[i]))
        ssim[i] = compare_ssim(slice_gt.astype(np.float64), slice_final.astype(np.float64), data_range=1)

    # Print results
    print("Mean (Standard deviation)")
    print("dist_err (mm) = {:.10f} ({:.10f})\n"
          "angle_err (deg) = {:.10f} ({:.10f})\n"
          "mse = {:.10f} ({:.10f})\n"
          "psnr = {:.10f} ({:.10f})\n"
          "ssim = {:.10f} ({:.10f})".format(np.mean(dist_err), np.std(dist_err),
                                            np.mean(angle_err), np.std(angle_err),
                                            np.mean(mse), np.std(mse),
                                            np.mean(psnr), np.std(psnr),
                                            np.mean(ssim), np.std(ssim)))

    return dist_err, angle_err, mse, psnr, ssim
