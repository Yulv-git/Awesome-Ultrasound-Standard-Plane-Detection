#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-19 10:33:38
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-23 00:32:15
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/ITN/utils/plane.py
Description: Functions for plane manipulations.
Init from https://github.com/yuanwei1989/plane-detection
'''
import numpy as np
import scipy.ndimage

from utils import geometry


def fit_plane(pts):
    """Fit a plane to a set of 3D points.
    Args:
      pts: [point_count, 3]
    Returns:
      n: normal vector of plane [3]
      c: centroid of plane [3]
    """
    c = pts.mean(axis=0)
    A = pts - c
    u, s, vh = np.linalg.svd(A)
    n = vh[-1, :]
    # ensure z-component of normal vector is always consistent (eg. positive)
    if n[2] < 0:
        n = -n
    return n, c


def project_on_plane(pts, n, c):
    """Project points onto a 2D plane.
    Args:
      pts: [point_count, 3]
      n: normal vector of plane [3]
      c: centroid of plane [3]
    Returns:
      pts_new: points projected onto the plane [point_count, 3]
    """
    t = (np.dot(c, n) - np.dot(pts, n)) / np.dot(n, n)
    pts_new = pts + np.matmul(np.expand_dims(t, axis=1), np.expand_dims(n, axis=0))
    return pts_new


def fit_line(pts):
    """Fit a line to a set of 3D points.
    Args:
      pts: [point_count, 3]
    Returns:
      d: direction vector of line [3]
      c: a point on the line [3]
    """
    c = pts.mean(axis=0)
    A = pts - c
    u, s, vh = np.linalg.svd(A)
    d = vh[0, :]
    # ensure x-component of direction vector is always consistent (eg. positive)
    if d[0] < 0:
        d = -d
    return d, c


def extract_tform(landmarks, plane_name):
    """Compute the transformation that maps the reference xy-plane at origin to the GT standard plane.
    Args:
      landmarks: [landmark_count, 3] where landmark_count=16
      plane_name: 'tv' or 'tc'
    Returns:
      trans_vec: translation vector [3]
      quat: quaternions [4]
      mat: 4x4 transformation matrix [4, 4]
    """
    if plane_name == 'tv':
        # Landmarks lying on the TV plane
        landmarks_plane = np.vstack((landmarks[1:8], landmarks[12:14]))

        # Compute transformation
        z_vec, p_plane = fit_plane(landmarks_plane)
        landmarks_plane_proj = project_on_plane(landmarks_plane, z_vec, p_plane)
        landmarks_line = landmarks_plane_proj[[0, 1, 2, 7, 8], :]
        x_vec, p_line = fit_line(landmarks_line)
        y_vec = geometry.unit_vector(np.cross(z_vec, x_vec))

        # 4x4 transformation matrix
        mat = np.eye(4)
        mat[:3, :3] = np.vstack((x_vec, y_vec, z_vec)).transpose()
        mat[:3, 3] = landmarks_plane_proj[0]
        # Quaternions and translation vector
        quat = geometry.quaternion_from_matrix(mat[:3, :3])
        trans_vec = mat[:3, 3]

    elif plane_name == 'tc':
        # Landmarks lying on the TC plane
        cr = landmarks[10]
        cl = landmarks[11]
        csp = landmarks[12]

        # Compute transformation
        csp_cl = cl - csp
        csp_cr = cr - csp
        z_vec = np.cross(csp_cl, csp_cr)
        z_vec = geometry.unit_vector(z_vec)
        cr_cl_mid = (cr + cl) / 2.0
        x_vec = geometry.unit_vector(cr_cl_mid - csp)
        y_vec = geometry.unit_vector(np.cross(z_vec, x_vec))

        # 4x4 transformation matrix
        mat = np.eye(4)
        mat[:3, :3] = np.vstack((x_vec, y_vec, z_vec)).transpose()
        mat[:3, 3] = (cr_cl_mid + csp) / 2.0
        # Quaternions and translation vector
        quat = geometry.quaternion_from_matrix(mat[:3, :3])
        trans_vec = mat[:3, 3]

    else:
        raise ValueError('Invalid plane name.')

    return trans_vec, quat, mat


def init_mesh(mesh_siz):
    """Initialise identity plane with a fixed size
        Args:
            mesh_siz: size of plane. Odd number only. [2]
        Returns:
            mesh: mesh coordinates of identity plane. [4, num_mesh_points]
    """
    mesh_r = (mesh_siz - 1) / 2
    x_lin = np.linspace(-mesh_r[0], mesh_r[0], mesh_siz[0])
    y_lin = np.linspace(-mesh_r[1], mesh_r[1], mesh_siz[1])
    xy_coords = np.meshgrid(y_lin, x_lin)
    xyz_coords = np.vstack([xy_coords[1].reshape(-1),
                            xy_coords[0].reshape(-1),
                            np.zeros(mesh_siz[0] * mesh_siz[1]),
                            np.ones(mesh_siz[0] * mesh_siz[1])])
    return xyz_coords


def init_mesh_by_plane(mesh_siz, normal):
    """Initialise identity plane with a fixed size. Either xy, xz or yz-plane
        Args:
            mesh_siz: size of plane. Odd number only. [2]
            normal: direction of normal vector of mesh. ('x', 'y', 'z')
        Returns:
            mesh: mesh coordinates of identity plane. [4, num_mesh_points]
    """
    mesh_r = (mesh_siz - 1) / 2
    x_lin = np.linspace(-mesh_r[0], mesh_r[0], mesh_siz[0])
    y_lin = np.linspace(-mesh_r[1], mesh_r[1], mesh_siz[1])
    xy_coords = np.meshgrid(y_lin, x_lin)
    if normal=='z':
        xyz_coords = np.vstack([xy_coords[1].reshape(-1),
                                xy_coords[0].reshape(-1),
                                np.zeros(mesh_siz[0] * mesh_siz[1]),
                                np.ones(mesh_siz[0] * mesh_siz[1])])
    elif normal=='y':
        xyz_coords = np.vstack([xy_coords[1].reshape(-1),
                                np.zeros(mesh_siz[0] * mesh_siz[1]),
                                xy_coords[0].reshape(-1),
                                np.ones(mesh_siz[0] * mesh_siz[1])])
    elif normal=='x':
        xyz_coords = np.vstack([np.zeros(mesh_siz[0] * mesh_siz[1]),
                                xy_coords[1].reshape(-1),
                                xy_coords[0].reshape(-1),
                                np.ones(mesh_siz[0] * mesh_siz[1])])
    return xyz_coords


def init_mesh_ortho(mesh_siz):
    """Initialise identity plane with a fixed size. Either xy, xz or yz-plane
        Args:
            mesh_siz: size of plane. Odd number only. [2]
        Returns:
            xyz_coords: mesh coordinates of xy, xz and yz plane. [3, 4, num_mesh_points]
    """
    xy = init_mesh_by_plane(mesh_siz, 'z')
    xz = init_mesh_by_plane(mesh_siz, 'y')
    yz = init_mesh_by_plane(mesh_siz, 'x')
    xyz_coords = np.stack((xy, xz, yz), axis=0)
    return xyz_coords


def extract_plane_from_mesh(image, mesh, mesh_siz, order):
    """Extract a 2D plane image from the 3D volume given the mesh coordinates of the plane.
        Args:
          image: 3D volume. [x,y,z]
          mesh: mesh coordinates of a plane. [4, num_mesh_points]. Origin at volume centre
          mesh_siz: size of mesh [2]
          order: interpolation order (0-5)
        Returns:
          slice: 2D plane image [plane_siz[0], plane_siz[1]]
          new_coords: mesh coordinates of the plane. Origin at volume corner. numpy array of size [3, plane_siz[0], plane_siz[1]
    """
    # Set image matrix corner as origin
    img_siz = np.array(image.shape)
    img_c = (img_siz-1)/2.0
    mesh_new = mesh[:3, :] + np.expand_dims(img_c, axis=1)

    # Reshape coordinates
    x_coords = mesh_new[0, :].reshape(mesh_siz)
    y_coords = mesh_new[1, :].reshape(mesh_siz)
    z_coords = mesh_new[2, :].reshape(mesh_siz)
    new_coords = np.stack((x_coords, y_coords, z_coords), axis=0)

    # Extract image plane
    slice = scipy.ndimage.map_coordinates(image, new_coords, order=order)
    return slice, new_coords


def extract_plane_from_mesh_batch(image, meshes, mesh_siz, order):
    """Extract a 2D plane image from the 3D volume given the mesh coordinates of the plane. Do it in a batch of planes
        Args:
          image: 3D volume. [x,y,z]
          meshes: mesh coordinates of planes. [mesh_count, 4, num_mesh_points]. Origin at volume centre
          mesh_siz: size of mesh [2]
          order: interpolation order (0-5)
        Returns:
          slices: 2D plane images [mesh_count, plane_siz[0], plane_siz[1]]
          meshes_new: mesh coordinates of the plane. Origin at volume corner. numpy array of size [mesh_count, plane_siz[0], plane_siz[1], 3]
    """
    # Set image matrix corner as origin
    img_siz = np.array(image.shape)
    img_c = (img_siz-1)/2.0
    mesh_count = meshes.shape[0]
    meshes_new = meshes[:, :3, :] + img_c[np.newaxis, :, np.newaxis]    # meshes_new = [mesh_count, 4, num_mesh_pts]
    meshes_new = np.reshape(np.transpose(meshes_new, (1, 0, 2))[:3], (3, mesh_count, mesh_siz[0], mesh_siz[1]))      # [3, mesh_count, plane_siz[0], plane_siz[1]]

    # Extract image plane
    slices = scipy.ndimage.map_coordinates(image, meshes_new, order=order)
    meshes_new = np.transpose(meshes_new, (1, 2, 3, 0))
    return slices, meshes_new


def extract_plane_from_mesh_ortho_batch(image, meshes, mesh_siz, order):
    """Extract orthogonal 2D plane images from the 3D volume given the mesh coordinates of the plane. Do it in a batch of planes
        Args:
          image: 3D volume. [x,y,z]
          meshes: mesh coordinates of planes. [mesh_count, mesh_ind, 4, num_mesh_pts]. Origin at volume centre
          mesh_siz: size of mesh [2]
          order: interpolation order (0-5)
        Returns:
          slices: 2D plane images [mesh_count, mesh_ind, plane_siz[0], plane_siz[1]]
          meshes_new: mesh coordinates of the plane. Origin at volume corner. numpy array of size [mesh_count, mesh_ind, plane_siz[0], plane_siz[1], 3]
    """
    # Set image matrix corner as origin
    input_plane = meshes.shape[1]
    img_siz = np.array(image.shape)
    img_c = (img_siz-1)/2.0
    mesh_count = meshes.shape[0]
    meshes_new = meshes[:, :, :3, :] + img_c[np.newaxis, :, np.newaxis]    # meshes_new = [mesh_count, mesh_ind, 4, num_mesh_pts]
    meshes_new = np.reshape(np.transpose(meshes_new, (2, 0, 1, 3))[:3], (3, mesh_count, input_plane, mesh_siz[0], mesh_siz[1]))      # [3, mesh_count, mesh_ind, plane_siz[0], plane_siz[1]]

    # Extract image plane
    slices = scipy.ndimage.map_coordinates(image, meshes_new, order=order)
    meshes_new = np.transpose(meshes_new, (1, 2, 3, 4, 0))
    return slices, meshes_new


def extract_plane_from_pose(image, t, q, plane_siz, order):
    """Extract a 2D plane image from the 3D volume given the pose wrt the identity plane.
        Args:
          image: 3D volume. [x,y,z]
          t: translation of the pose [3]
          q: rotation of the pose in quaternions [4]
          plane_siz: size of plane [2]
          order: interpolation order (0-5)
        Returns:
          slice: 2D plane image [plane_siz[0], plane_siz[1]]
          mesh: mesh coordinates of the plane. Origin at volume corner. numpy array of size [3, plane_siz[0], plane_siz[1]]
    """
    # Initialise identity plane
    xyz_coords = init_mesh(plane_siz)

    # Rotate and translate plane
    mat = geometry.quaternion_matrix(q)
    mat[:3, 3] = t
    xyz_coords = np.dot(mat, xyz_coords)

    # Extract image plane
    slice, xyz_coords_new = extract_plane_from_mesh(image, xyz_coords, plane_siz, order)

    return slice, xyz_coords_new
