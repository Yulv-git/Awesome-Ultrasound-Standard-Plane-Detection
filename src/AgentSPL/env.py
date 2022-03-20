# -*- coding: utf-8 -*-
"""
This is to define the environment in the reinforcement learning.
"""
import random
import copy
import os

import gym
import vtk
import numpy as np
from gym.utils import seeding

from utils import get_plane


class USVolumeEnv(gym.Env):
    def __init__(self, path, option, is_train):
        """
        init the environment
        :param path: volume and annotation path
        :param option: option about the environment

        ==============================
        option including as follow:
        annotation: whether load the mean annotation or once, [First, Second, Mean], string
        target: the id of the planes, [1, 2, 3, 4], string
        env_init: mode of the init plane parameter, [Random, ], string
        ==============================
        """
        self.__version__ = "0.1.0"

        # define the termination signal
        self.termination = False

        self.path = path
        self.option = option
        self.is_train = is_train

        # loading data and annotation file
        self.volume, self.target_plane, self.start_plane = self.load_data()

        # get the angle in each axes
        self.start_x = np.rad2deg(np.arccos(self.start_plane["normal"][0]))
        self.start_y = np.rad2deg(np.arccos(self.start_plane["normal"][1]))
        self.start_z = np.rad2deg(np.arccos(self.start_plane["normal"][2]))

        # copy the value of the start angle to the current angle
        self.current_x = copy.copy(self.start_x)
        self.current_y = copy.copy(self.start_y)
        self.current_z = copy.copy(self.start_z)

        # copy the the start plane parameter to the current parameter
        self.current_plane = copy.copy(self.start_plane)

        # define the step angle and step distance
        self.step_angle = 1.0
        self.step_dis = 0.5

        self.np_random = None
        self.seed()

        self.d = 0

    def step(self, action):
        """
        step function of the env
        """
        # take the action and change the plane parameter
        self.take_action(action)

        # get the reward
        reward = self._get_reward()

        # get the next state(obj)
        obj = self.get_state()

        # calculate the angle and distance between the current and target
        angle, distance = self.metric_calculate()

        info = {
            "current_plane": self.current_plane,
            "reward": reward,
            "distance": distance,
            "angle": angle,
            "termination": self.termination
        }
        return obj, reward, self.termination, info

    def take_action(self, action):
        ex_normal_x, ex_normal_y, ex_normal_z = self.current_plane["normal"]
        ex_p = self.current_plane["p"]

        # get the ex plane parameter
        ex_arr = np.array([ex_normal_x, ex_normal_y, ex_normal_z, ex_p], dtype=np.float32)

        # make the action to change.
        delta_p = 0

        if action == 0:
            self.current_x += self.step_angle
        elif action == 1:
            self.current_y += self.step_angle
        elif action == 2:
            self.current_z += self.step_angle
        elif action == 3:
            delta_p = self.step_dis
        elif action == 4:
            self.current_x -= self.step_angle
        elif action == 5:
            self.current_y -= self.step_angle
        elif action == 6:
            self.current_z -= self.step_angle
        elif action == 7:
            delta_p = -self.step_dis
        else:
            raise ValueError

        # get the new normal from the angle
        current_normal = [
            np.cos(np.deg2rad(self.current_x)),
            np.cos(np.deg2rad(self.current_y)),
            np.cos(np.deg2rad(self.current_z)),
        ]

        current_normal_arr = np.array(current_normal, dtype=np.float32)
        current_p = delta_p + ex_p

        # update the current
        self.current_plane["normal"] = current_normal_arr
        self.current_plane["p"] = current_p

        current = [current_normal[0], current_normal[1], current_normal[2], current_p]
        target = [self.target_plane["normal"][0],
                  self.target_plane["normal"][1],
                  self.target_plane["normal"][2],
                  self.target_plane["p"]]

        current_arr = np.array(current, dtype=np.float32)
        target_arr = np.array(target, dtype=np.float32)

        # calculate the reward by calculate the ex and current
        sub_ex_target = ex_arr - target_arr
        sub_current_target = current_arr - target_arr

        # the d is to calculate for the reward function
        self.d = np.linalg.norm(sub_ex_target) - np.linalg.norm(sub_current_target)

    def _get_reward(self):
        """
        the reward is a sign function for the d
        :return:
        """
        reward = np.sign(self.d)
        return reward

    def get_state(self):
        """
        get the current plane data
        :return:
        """
        current_plane = get_plane(reader=self.volume,
                                  plane_parameter=self.current_plane)
        return current_plane

    def reset(self):
        """
        reset the Volume Environment
        :return:
        """
        # recopy the angle
        self.current_x = copy.copy(self.start_x)
        self.current_y = copy.copy(self.start_y)
        self.current_z = copy.copy(self.start_z)

        # recopy the plane
        self.current_plane = copy.copy(self.start_plane)

        # set the termination signal to be false
        self.termination = False

        # get the start plane data
        start_plane = self.get_state()

        return start_plane

    def render(self, mode='human', close=False):
        """
        I don not know the usage, but it seem not influence the learning process
        :param mode:
        :param close:
        :return:
        """
        return

    def seed(self, seed=None):
        """
        I don not know the usage, but it seem not influence the learning process
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_data(self):
        """
        Loading the volume and annotation
        :return:
        """
        # load the volume
        volume_path = os.path.join(self.path, "data.nii.gz")

        # loading the volume based the vtk reader
        nii_reader = vtk.vtkNIFTIImageReader()
        nii_reader.SetFileName(volume_path)
        nii_reader.Update()

        # change the information of the nii reader
        volume_reader = vtk.vtkImageChangeInformation()
        volume_reader.SetInputData(nii_reader.GetOutput())
        volume_reader.SetOutputSpacing(1.0, 1.0, 1.0)
        # noting that we make center of the volume as the origin in axes
        volume_reader.CenterImageOn()
        volume_reader.Update()

        # load the annotation file
        info = np.load(os.path.join(self.path, "info.npz"), encoding="bytes", allow_pickle=True)
        target_info = info['data'][()]
        target_normal = np.squeeze(target_info['normal'])
        target_center = np.squeeze(target_info['center'])

        target_p = np.dot(target_normal, target_center)
        # if target_p < 0:
        #     target_p *= -1
        #     target_normal *= -1
        # else:
        #     pass
        target_plane = {
            "normal": target_normal,
            "p": target_p
        }

        if self.is_train:
            x, y, z = target_normal

            angle_x = np.rad2deg(np.arccos(x)) + 25 - 50 * random.random()  # 35-70
            angle_y = np.rad2deg(np.arccos(y)) + 25 - 50 * random.random()
            angle_z = np.rad2deg(np.arccos(z)) + 25 - 50 * random.random()

            cos_x = np.cos(np.deg2rad(angle_x))
            cos_y = np.cos(np.deg2rad(angle_y))
            cos_z = np.cos(np.deg2rad(angle_z))
            normal = np.array([cos_x, cos_y, cos_z], dtype=np.float32)
            p = np.array([np.dot(target_normal, target_center)], dtype=np.float32)
            if p < 0:
                p *= -1
                normal *= -1
            p += 15 * random.random()
        else:
            atlas = np.load(os.path.join(self.option.data_path, "atlas/info.npz"), encoding="bytes", allow_pickle=True)
            # atlas = np.load(os.path.join(self.option.data_path, "test_atlas/info.npz"), encoding="bytes")
            atlas_info = atlas["data"][()]
            start_normal = atlas_info["normal"]
            start_center = atlas_info["center"]

            x, y, z = start_normal

            angle_x = np.rad2deg(np.arccos(x))
            angle_y = np.rad2deg(np.arccos(y))
            angle_z = np.rad2deg(np.arccos(z))

            cos_x = np.cos(np.deg2rad(angle_x))
            cos_y = np.cos(np.deg2rad(angle_y))
            cos_z = np.cos(np.deg2rad(angle_z))

            normal = np.array([cos_x, cos_y, cos_z], dtype=np.float32)
            p = np.array([np.dot(start_normal, start_center)], dtype=np.float32)
            # if p < 0:
            #     p *= -1
            #     normal *= -1
        start_plane = {"normal": normal, "p": p}

        return volume_reader, target_plane, start_plane

    def metric_calculate(self):
        """
        calculate the distance and angle between the current plane and target plane
        """
        current_normal, current_p = self.current_plane["normal"], self.current_plane["p"]
        target_normal, target_p = self.target_plane["normal"], self.target_plane["p"]

        current_l = np.sqrt(current_normal.dot(current_normal))
        target_l = np.sqrt(target_normal.dot(target_normal))

        cos_angle = current_normal.dot(target_normal)/(current_l*target_l)

        rad_angle = np.arccos(cos_angle)
        deg_angle = np.rad2deg(rad_angle)

        if deg_angle > 90:
            deg_angle = 180 - deg_angle

        distance = np.abs(target_p - current_p)

        return float(deg_angle), float(distance)

