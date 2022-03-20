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
    class Parser(object):
        """
        define the option of the training
        """
        def __init__(self):
            # =============== define environment

            # =============== define training
            # batch size, INT
            self.batch_size = 4
            # target net weight update term, INT
            self.target_step_counter = 1500
            # learning rate, FLOAT
            self.lr = 5e-5
            # weight decay, FLOAT
            self.weight_decay = 1e-4
            # reward decay, FLOAT
            self.gamma = 0.95
            # memory capacity, INT
            self.memory_capacity = 15000
            # epsilon for the greedy, FLOAT
            self.epsilon = 0.6

            # =============== define default
            # gpu id
            self.gpu_id = 0
            # total epoch
            self.num_epoch = 100
            # max steps
            self.max_step = 75

            # =============== define path
            # data path
            self.data_path = "template_data/subjects"
            self.output_path = "output"
            self.list_path = "template_data"

    parser = Parser()

    main(parser)
