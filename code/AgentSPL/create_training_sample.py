#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-20 18:17:37
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-03 17:15:37
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/AgentSPL/create_training_sample.py
Description: Modify here please
Init from https://github.com/wulalago/AgentSPL
'''
import argparse
from itertools import count
import os
import copy
import torch
import numpy as np

from env import USVolumeEnv
from agent import Agent
from utils import read_list, image_process, check_dir, AvgMeter


def main(args):
    torch.cuda.set_device(args.gpu_id)

    data_list = read_list(args.list_path, args.data_path, mode="train")

    check_dir(os.path.join(args.output_path, "RNN_training_samples"))

    agent = Agent(args)
    agent.eval_net.load_state_dict(torch.load(os.path.join(args.output_path, "checkpoints", "best_adi.pth.gz")))

    dis_avg = AvgMeter()
    ang_avg = AvgMeter()
    adi_avg = AvgMeter()

    start_ang_list = []
    start_dis_list = []
    max_dis_list = []
    max_ang_list = []
    max_adi_list = []
    ter_dis_list = []
    ter_ang_list = []
    ter_adi_list = []
    lim_adi_list = []
    lim_ang_list = []
    lim_dis_list = []

    data_list.sort()
    for train_path in data_list:
        train_id = train_path.split("\\")[-1]
        env = USVolumeEnv(train_path, args, is_train=False)

        start_angle, start_distance = env.metric_calculate()

        start_ang_list.append(float(start_angle))
        start_dis_list.append(float(start_distance))

        current_state_plane = image_process(env.get_state())
        state = np.concatenate((current_state_plane, current_state_plane, current_state_plane), axis=0)

        distance_list = []
        angle_list = []
        adi_list = []
        t_list = []
        q_list = []

        test_value_list = []
        test_plane_list = []

        for t in count(1):
            t_list.append(t)

            # ================== select action =============
            action, q_values = agent.select_action(state, return_q=True, random_action=False)

            test_value_list.append(q_values[np.newaxis, :])

            plane = np.concatenate((env.current_plane["normal"], env.current_plane["p"]))[np.newaxis, :]

            test_plane_list.append(plane)

            q_values = np.squeeze(q_values)
            q_value = np.mean(q_values)
            q_list.append(float(q_value))
            # ================== select action =============

            next_state_plane, reward, termination, info_log = env.step(action)

            next_state = np.concatenate((state[1:, :, :], image_process(next_state_plane)))

            state = copy.copy(next_state)

            adi = start_angle + start_distance - info_log["distance"] - info_log["angle"]

            angle_list.append(float(info_log["angle"]))
            distance_list.append(float(info_log["distance"]))
            adi_list.append(float(adi))

            if t % 1 == 0:
                print("\r {}--{} | Distance: {:.2f} | Angle: {:.2f} | Action: {}".format(
                    train_id, t, info_log["distance"], info_log["angle"], action), end="")

            if t == args.max_step:
                dis_avg.update(info_log["distance"])
                ang_avg.update(info_log["angle"])
                adi_avg.update(adi)

                max_dis_list.append(float(info_log["distance"]))
                max_ang_list.append(float(info_log["angle"]))
                max_adi_list.append(float(adi))
                info = [str(train_id).zfill(3), t, info_log["distance"], info_log["angle"], adi]
                print("\rTest ID: {} || Step: {} Dis: {:.4f} Angle: {:.4f} | yes:{:+.2f}".format(*info))

                ind = q_list.index(min(q_list))
                # print(yes_list[ind])
                ter_adi_list.append(adi_list[ind])
                ter_ang_list.append(angle_list[ind])
                ter_dis_list.append(distance_list[ind])

                ind_ = adi_list.index(max(adi_list))
                lim_adi_list.append(adi_list[ind_])
                lim_ang_list.append(angle_list[ind_])
                lim_dis_list.append(distance_list[ind_])

                train_value_seq = np.concatenate(test_value_list, axis=0)
                train_plane_seq = np.concatenate(test_plane_list, axis=0)
                np.save(os.path.join(args.output_path, "RNN_training_samples", "{}_q.npy".format(train_id)), train_value_seq)
                np.save(os.path.join(args.output_path, "RNN_training_samples", "{}_angle.npy".format(train_id)),
                        np.array(angle_list, dtype=np.float32))
                np.save(os.path.join(args.output_path, "RNN_training_samples", "{}_distance.npy".format(train_id)),
                        np.array(distance_list, dtype=np.float32))
                np.save(os.path.join(args.output_path, "RNN_training_samples", "{}_yes.npy".format(train_id)),
                        np.array(adi_list, dtype=np.float32))
                np.save(os.path.join(args.output_path, "RNN_training_samples", "{}_plane.npy".format(train_id)), train_plane_seq)

                break


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Generate the training samples for RNN(termination)')
    # define path
    parse.add_argument('--data_path', type=str, default='./template_data/subjects', help="data path")
    parse.add_argument('--output_path', type=str, default='./output', help="")
    parse.add_argument('--list_path', type=str, default='./template_data', help="")
    # define default
    parse.add_argument('--gpu_id', type=int, default=0, help="gpu id")
    parse.add_argument('--num_epoch', type=int, default=100, help="total epoch")
    parse.add_argument('--max_step', type=int, default=75, help="max steps")
    # define training
    parse.add_argument('--batch_size', type=int, default=4, help="batch size")
    parse.add_argument('--target_step_counter', type=int, default=1500, help="target net weight update term")
    parse.add_argument('--lr', type=float, default=5e-5, help="weight decay")
    parse.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parse.add_argument('--gamma', type=float, default=0.95, help="reward decay")
    parse.add_argument('--memory_capacity', type=int, default=15000, help="memory capacity")
    parse.add_argument('--epsilon', type=float, default=0.6, help="epsilon for the greedy")
    args = parse.parse_args()

    main(args)
