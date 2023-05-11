#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-19 10:33:38
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-23 14:37:32
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/ITN/utils/visual.py
Description: Functions for visualisations.
Init from https://github.com/yuanwei1989/plane-detection
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from utils import geometry


def plot_planes(save_dir, suffix, name, img_siz, trans_vec_final, quat_final, mesh_final, trans_vec_gt, quat_gt, mesh_gt):
    """Plot GT and predicted planes"""
    img_c = (img_siz-1)/2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf_gt = ax.plot_surface(mesh_gt[1], mesh_gt[0], mesh_gt[2])
    surf_gt.set_facecolor([1, 0, 0, 0.3])
    mat_gt = geometry.quaternion_matrix(quat_gt)
    ax.quiver(trans_vec_gt[1] + img_c[1], trans_vec_gt[0] + img_c[0], trans_vec_gt[2] + img_c[2], mat_gt[:, 1],
              mat_gt[:, 0], mat_gt[:, 2], length=30, color='r')

    surf_final = ax.plot_surface(mesh_final[1], mesh_final[0], mesh_final[2])
    surf_final.set_facecolor([0, 1, 0, 0.3])
    mat_final = geometry.quaternion_matrix(quat_final)
    ax.quiver(trans_vec_final[1] + img_c[1], trans_vec_final[0] + img_c[0], trans_vec_final[2] + img_c[2],
              mat_final[:, 1], mat_final[:, 0], mat_final[:, 2], length=30, color='g')

    # plt.axis('equal')
    plt.axis('auto')
    ax.set_title('{0}'.format(name))
    ax.set_xlim(0, img_siz[1])
    ax.set_ylim(0, img_siz[0])
    ax.set_zlim(0, img_siz[2])
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_zlabel('z')
    ax.view_init(elev=8., azim=63)
    ax.invert_xaxis()

    save_dir = os.path.join(save_dir, suffix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, name + '.png'), bbox_inches='tight')
    plt.close(fig)


def plot_images(save_dir, suffix, name, slice_final, slice_gt):
    """Plot GT and predicted plane images"""
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(slice_gt, cmap='gray')
    plt.title('GT plane')
    plt.subplot(122)
    plt.imshow(slice_final, cmap='gray')
    plt.title('Predicted plane')
    save_dir = os.path.join(save_dir, suffix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, name + '.png'), bbox_inches='tight')
    plt.close(fig)


def plot_planes_movie(save_dir, suffix, name, img_siz, max_test_steps, meshes, matrices, trans_vec_gt, quat_gt, mesh_gt):
    """Save animation of planes over several test iterations"""
    img_c = (img_siz-1)/2
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    def update_plane(n):
        ax1.cla()
        fig1.set_tight_layout(True)

        # GT plane
        surf_gt = ax1.plot_surface(mesh_gt[1], mesh_gt[0], mesh_gt[2])
        surf_gt.set_facecolor([1, 0, 0, 0.3])
        mat_gt = geometry.quaternion_matrix(quat_gt)
        quiv_gt = ax1.quiver(trans_vec_gt[1] + img_c[1], trans_vec_gt[0] + img_c[0], trans_vec_gt[2] + img_c[2],
                             mat_gt[:, 1], mat_gt[:, 0], mat_gt[:, 2], length=30, color='r')

        # Predicted planes
        surf = ax1.plot_surface(meshes[0, n, 0, :, :, 1], meshes[0, n, 0, :, :, 0], meshes[0, n, 0, :, :, 2])
        surf.set_facecolor([0, 1, 0, 0.3])
        quiv = ax1.quiver(matrices[0, n, 1, 3] + img_c[1], matrices[0, n, 0, 3] + img_c[0],
                          matrices[0, n, 2, 3] + img_c[2], matrices[0, n, :, 1], matrices[0, n, :, 0],
                          matrices[0, n, :, 2], length=30, color='g')
        # plt.axis('equal')
        plt.axis('auto')
        ax1.set_title('iteration {}'.format(n))
        ax1.set_xlim(0, img_siz[1])
        ax1.set_ylim(0, img_siz[0])
        ax1.set_zlim(0, img_siz[2])
        ax1.set_xlabel('y')
        ax1.set_ylabel('x')
        ax1.set_zlabel('z')
        ax1.view_init(elev=8., azim=63)
        ax1.invert_xaxis()

        return surf_gt, quiv_gt, surf, quiv

    anim = FuncAnimation(fig1, update_plane,
                         frames=np.arange(0, max_test_steps + 1, 1),
                         interval=400,
                         repeat_delay=3000,
                         repeat=True)
    save_dir = os.path.join(save_dir, suffix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    anim.save(os.path.join(save_dir, name + '.gif'), dpi=100, writer='imagemagick')
    plt.close(fig1)


def plot_images_movie(save_dir, suffix, name, max_test_steps, slices, slice_gt):
    """Save animation of plane images over several test iterations"""
    fig2 = plt.figure()
    ax2_1 = fig2.add_subplot(121)
    ax2_2 = fig2.add_subplot(122)

    def update_image(n):
        ax2_1.cla()
        ax2_2.cla()
        fig2.set_tight_layout(True)

        # GT image
        ax2_1.axis('off')
        img_obj_gt = ax2_1.imshow(slice_gt, cmap='gray')
        ax2_1.set_title('GT plane')

        # predicted image
        img_obj = ax2_2.imshow(slices[0, n, 0], cmap='gray')
        ax2_2.axis('off')
        ax2_2.set_title('Iteration {}\nPredicted plane'.format(n))

        return img_obj_gt, img_obj

    anim = FuncAnimation(fig2, update_image,
                         frames=np.arange(0, max_test_steps + 1, 1),
                         interval=400,
                         repeat_delay=3000,
                         repeat=True)
    save_dir = os.path.join(save_dir, suffix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    anim.save(os.path.join(save_dir, name + '.gif'), dpi=100, writer='imagemagick')
    plt.close(fig2)
