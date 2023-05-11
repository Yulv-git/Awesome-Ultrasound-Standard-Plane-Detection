#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-19 10:33:38
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-23 12:43:29
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/ITN/utils/save.py
Description: Functions for writing results.
Init from https://github.com/yuanwei1989/plane-detection
'''
import os
import numpy as np
import nibabel as nib

from utils import geometry


def save_err(save_dir, suffix, names, dist_err, angle_err, mse, psnr, ssim):
    """Save the evaluation results in a txt file.
    Args:
      save_dir: Directory storing the results.
      suffix: 'train' or 'test'
      names: list of names of the images.
      dist_err, angle_err, mse, psnr, ssim: error metrics. [img_count]
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'eval_' + suffix + '.txt'), 'w') as f:
        f.write("dist_err (mm)\tangle_err (deg)\tmse\tpsnr\tssim\n")
        for i in range(len(names)):
            f.write("{} {:.10f} {:.10f} {:.10f} {:.10f} {:.10f}\n".format(names[i],
                                                                          dist_err[i],
                                                                          angle_err[i],
                                                                          mse[i],
                                                                          psnr[i],
                                                                          ssim[i]))
        f.write("\nMean (Standard deviation)\n")
        f.write("dist_err (mm) = {:.10f} ({:.10f})\n"
                "angle_err (deg) = {:.10f} ({:.10f})\n"
                "mse = {:.10f} ({:.10f})\n"
                "psnr = {:.10f} ({:.10f})\n"
                "ssim = {:.10f} ({:.10f})".format(np.mean(dist_err), np.std(dist_err),
                                                  np.mean(angle_err), np.std(angle_err),
                                                  np.mean(mse), np.std(mse),
                                                  np.mean(psnr), np.std(psnr),
                                                  np.mean(ssim), np.std(ssim)))


def save_planes_tform(save_dir, suffix, names, trans_vecs, quats):
    """Save the plane as a 4x4 transformation matrix in a txt file.
    Args:
      save_dir: Directory storing the results.
      suffix: 'train' or 'test'
      names: list of names of the patients.
      trans_vecs: translation vectors. [img_count, 3]
      quats: quaternions. [img_count, 4]
    """
    save_dir = os.path.join(save_dir, suffix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for i in range(len(names)):
        mat = geometry.quaternion_matrix(quats[i, :])
        mat[:3, 3] = trans_vecs[i, :]
        np.savetxt(os.path.join(save_dir, names[i] + '_mat.txt'), mat)


def save_planes_nifti(save_dir, suffix, name, slice_final, slice_gt):
    """Save plane image as a nifti file.
    Args:
      save_dir: Directory storing the nifti image.
      suffix: 'train' or 'test'
      names: list of names of the patients.
      slice_final: Predicted plane image.
      slice_gt: GT plane image.
    """
    save_path = os.path.join(save_dir, 'GT', suffix)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    img_gt = nib.Nifti1Image(slice_gt, np.eye(4))
    nib.save(img_gt, os.path.join(save_path, name + '.nii.gz'))
    save_path = os.path.join(save_dir, 'Predict', suffix)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    img_final = nib.Nifti1Image(slice_final, np.eye(4))
    nib.save(img_final, os.path.join(save_path, name + '.nii.gz'))
