import glob
import os

import numpy as np
import cv2
import vtk
import random
from matplotlib import pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy


def normalized_vector(vector):
    """
    normalized a vector
    :param vector: input vector
    :return: vector after normalization
    """
    vector = vector / np.linalg.norm(vector)
    return vector


def get_matrix(start_vector, end_vector):
    """
    this function is to get the rotation matrix between the start_vector and end vector
    both the two vector use the origin as the start point.
    :param start_vector: the start vector before rotate
    :param end_vector: the vector after rotate
    :return: the rotation matrix and quaternions
    """

    # transfer the vector to the numpy array
    start_vector = np.array(start_vector).astype(np.float32)
    end_vector = np.array(end_vector).astype(np.float32)

    # normalized the vector
    start_vector = normalized_vector(start_vector)
    end_vector = normalized_vector(end_vector)

    # get the axis vector
    axis_vector = np.cross(start_vector, end_vector)
    axis_vector = normalized_vector(axis_vector)

    # calculate the axis angle
    l_start_vector = np.sqrt(start_vector.dot(start_vector))
    l_end_vector = np.sqrt(end_vector.dot(end_vector))
    cos_angle = start_vector.dot(end_vector) / (l_start_vector * l_end_vector)
    angle = np.arccos(cos_angle)
    half_angle = angle / 2.

    # calculate the quaternions based on the axis vector and axis angle
    quaternions = np.array(
        [
            np.cos(half_angle),
            np.sin(half_angle) * axis_vector[0],
            np.sin(half_angle) * axis_vector[1],
            np.sin(half_angle) * axis_vector[2]
        ]
    )
    w, x, y, z = quaternions

    # calculate the rotation matrix based on the quaternions
    # rotation_matrix = np.array(
    #     [
    #         [1 - 2 * y * y - 2 * z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
    #         [2 * (x * y + z * w), 1 - 2 * x * x - 2 * z * z, 2 * (y * z - x * w)],
    #         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * x * x - 2 * y * y],
    #     ]
    # )
    rotation_matrix = np.array([
        [w*w + x*x - y*y - z*z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), w*w - x*x + y*y - z*z, 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), w*w - x*x - y*y + z*z]
    ])
    return quaternions, rotation_matrix


def get_plane_from_matrix(reader, rotate_matrix, center):
    """
    this function is to get the plane based on the rotate matrix,
    it also need the input vtk reader and center point index after rotate
    :param reader: the vtk reader of the volume
    :param rotate_matrix: the rotation matrix of normal vector started from the z-normal (0, 0, 1)
    :param center: the point in the rotated plane
    :return: the numpy data of the plane
    """
    # define the matrix copy to the vtk slice-axes
    copy_matrix = (rotate_matrix[0, 0], rotate_matrix[0, 1], rotate_matrix[0, 2], center[0],
                   rotate_matrix[1, 0], rotate_matrix[1, 1], rotate_matrix[1, 2], center[1],
                   rotate_matrix[2, 0], rotate_matrix[2, 1], rotate_matrix[2, 2], center[2],
                   0, 0, 0, 1)
    # copy_matrix = (rotate_matrix[0, 0], rotate_matrix[0, 1], rotate_matrix[0, 2], 0,
    #                rotate_matrix[1, 0], rotate_matrix[1, 1], rotate_matrix[1, 2], 0,
    #                rotate_matrix[2, 0], rotate_matrix[2, 1], rotate_matrix[2, 2], 0,
    #                0, 0, 0, 1)

    matrix = vtk.vtkMatrix4x4()
    matrix.DeepCopy(copy_matrix)

    ResliceTransform = vtk.vtkTransform()
    ResliceTransform.SetMatrix(matrix)
    # define the reslice class
    reslice = vtk.vtkImageReslice()
    reslice.SetInputConnection(reader.GetOutputPort())
    reslice.SetOutputDimensionality(3)
    # reslice.SetResliceAxes(matrix)
    reslice.SetResliceTransform(ResliceTransform)

    reslice.SetInterpolationModeToLinear()
    # reslice.AutoCropOutputOn()
    reslice.Update()
    reslice.SetOutputSpacing(1.0, 1.0, 1.0)
    origin_size = reslice.GetOutput().GetDimensions()
    reslice.SetOutputExtent(0, int(origin_size[0]), 0, int(origin_size[1]), 0, 0)

    reslice.Update()

    # write the tif image to the local path
    # writer = vtk.vtkTIFFWriter()
    # writer.SetInputConnection(reslice.GetOutputPort())
    # writer.SetFileName("temp.tif")
    # writer.Write()

    # transform the plane from the reslice to the numpy data
    numpy_data = get_numpy_from_reslice(reslice)
    return numpy_data


def get_numpy_from_reslice(reslice):
    reslice_image = reslice.GetOutput()
    output = reslice_image.GetPointData().GetScalars()
    dims = reslice_image.GetDimensions()
    numpy_data = vtk_to_numpy(output)
    numpy_data = numpy_data.reshape(dims[2], dims[1], dims[0])
    numpy_data = numpy_data.transpose(2, 1, 0)
    numpy_data = np.squeeze(numpy_data)
    return numpy_data


def get_plane(reader, plane_parameter):
    """
    this function is to get the plane from the reader based on the plane parameter and used to the env
    :param reader: vtk nii reader
    :param plane_parameter: dict include normal and p
    :return: plane numpy data
    """
    plane_normal = plane_parameter["normal"]
    plane_p = plane_parameter["p"]

    # here we let the plane normal not equal to 0,0,1, it may occur a bug when normal is 0,0,1
    if plane_normal[0] == 0 and plane_normal[1] == 0 and plane_normal[2] == 1:
        plane_normal = np.array([1e-5, 1e-5, 1+2e-5], dtype=np.float32)

    # we let the point be the closet point to the origin in the plane to ensure the image fully got.
    a, b, c = plane_normal
    k = -2*plane_p / (a*a+b*b+c*c)

    point = np.array([-0.5*a*k, -0.5*b*k, -0.5*c*k], dtype=np.float32)

    # we set the default vector before rotate as z unit normal vector
    z_normal = np.array([0, 0, 1], dtype=np.float32)

    _, rotate_mat = get_matrix(z_normal, plane_normal)

    plane = get_plane_from_matrix(reader=reader, rotate_matrix=rotate_mat, center=point)

    return plane


def read_list(txt_path, data_path, mode):
    if mode == "train":
        path = os.path.join(txt_path, "list_train.txt")
    elif mode == "val":
        path = os.path.join(txt_path, "list_val.txt")
    elif mode == "test":
        path = os.path.join(txt_path, "list_test.txt")
    else:
        raise NameError
    fid = open(path, 'r')
    reader_list = fid.readlines()
    reader_list = [os.path.join(data_path, line.split("\n")[0]) for line in reader_list]
    fid.close()
    random.shuffle(reader_list)
    return reader_list


class AvgMeter(object):
    """
    this class is to record one variable such as loss or acc
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def image_process(image, desired_size=224):
    """
    The pre process function to input image to the network
    """
    image = pad_to_desire_size(image, desired_size=desired_size)
    image = image[np.newaxis, :, :].astype(np.float32)
    image = image / 255.
    return image


def pad_to_desire_size(x, desired_size=224):
    """
    this function is to pad the image to the desired_size,
    1) resize
    2) pad
    :param x: image
    :param desired_size: default is 224
    :return:
    """

    old_size = x.shape[:2]
    # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    x = cv2.resize(x, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]

    x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return x


def plot(data, output_path, y_label, color='r'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data, color, label=y_label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(output_path, "{}.png".format(y_label)))
    plt.close()


class PlotDriver(object):
    """
    This is the class to plot the figure of training curve
    """
    def __init__(self, output_path, colors, labels):
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.e = []
        self.f = []
        self.g = []

        self.output_path = output_path
        self.colors = colors
        self.labels = labels

    def plot(self):
        for data, color, label in zip([self.a, self.b, self.c, self.d, self.e, self.f, self.g], self.colors, self.labels):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(data, color, label=label)
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
            plt.legend()
            plt.savefig(os.path.join(self.output_path, "{}.png".format(label)))
            plt.close()

    def update(self, data1, data2, data3, data4, data5, data6, data7):
        self.a.append(data1)
        self.b.append(data2)
        self.c.append(data3)
        self.d.append(data4)
        self.e.append(data5)
        self.f.append(data6)
        self.g.append(data7)


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


