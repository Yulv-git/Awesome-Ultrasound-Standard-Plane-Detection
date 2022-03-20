import os
import copy
from itertools import count

import torch
import numpy as np

from env import USVolumeEnv
from agent import Agent, beta_by_frame
from utils import read_list, AvgMeter, image_process, PlotDriver, check_dir


def train_epoch(train_list, args, agent):
    epoch_loss = AvgMeter()
    epoch_angle = AvgMeter()
    epoch_distance = AvgMeter()
    epoch_adi = AvgMeter()

    for train_id, train_path in enumerate(train_list):
        # define the environment
        env = USVolumeEnv(train_path, args, is_train=True)
        # get the start angle and distance
        start_angle, start_distance = env.metric_calculate()

        # continuity:
        current_state_plane = image_process(env.get_state())
        state = np.concatenate((current_state_plane, current_state_plane, current_state_plane), axis=0)

        for t in count(1):
            agent.frame_idx += 1

            action = agent.select_action(state, return_q=False, random_action=True)

            next_state_plane, reward, termination, info_log = env.step(action)

            next_state = np.concatenate((state[1:, :, :], image_process(next_state_plane)))

            agent.replay_buffer.push(state, action, reward, next_state, termination)

            state = copy.copy(next_state)

            beta = beta_by_frame(agent.frame_idx)
            loss = agent.learn(beta)

            epoch_loss.update(loss)

            if t % 5 == 0:
                print("\r{}-{} | Loss: {:.4f} Reward: {:+} | Distance: {:.2f} Angle: {:.2f}".format(
                    train_id, t, loss, info_log["reward"], info_log["distance"], info_log["angle"]), end="")

            if t == args.max_step:
                adi = start_angle + start_distance - info_log["distance"] - info_log["angle"]

                info = [str(train_id).zfill(3), info_log["distance"], info_log["angle"], adi]
                print("\rTrain-{} || Dis: {:.2f} Angle: {:.2f} ADI: {:+.2f}".format(
                    *info))

                epoch_angle.update(info_log["angle"])
                epoch_distance.update(info_log["distance"])
                epoch_adi.update(adi)

                break
    return epoch_loss.avg, epoch_angle.avg, epoch_distance.avg, epoch_adi.avg


def val_epoch(val_list, args, agent):
    epoch_angle = AvgMeter()
    epoch_distance = AvgMeter()
    epoch_adi = AvgMeter()

    for val_id, val_path in enumerate(val_list[:20]):
        # define the environment
        env = USVolumeEnv(val_path, args, is_train=False)
        # get the start angle and distance
        start_angle, start_distance = env.metric_calculate()

        # continuity:
        current_state_plane = image_process(env.get_state())
        state = np.concatenate((current_state_plane, current_state_plane, current_state_plane), axis=0)

        for t in count(1):
            action = agent.select_action(state, return_q=False, random_action=False)

            next_state_plane, reward, termination, info_log = env.step(action)

            next_state = np.concatenate((state[1:, :, :], image_process(next_state_plane)))

            state = copy.copy(next_state)

            if t == args.max_step:
                adi = start_angle + start_distance - info_log["distance"] - info_log["angle"]

                info = [str(val_id).zfill(3), info_log["distance"], info_log["angle"], adi]
                print("\rVal-{} || Dis: {:.2f} Angle: {:.2f} ADI: {:+.2f}".format(
                    *info))

                epoch_angle.update(info_log["angle"])
                epoch_distance.update(info_log["distance"])
                epoch_adi.update(adi)

                break
    return epoch_angle.avg, epoch_distance.avg, epoch_adi.avg


def main(args):
    torch.cuda.set_device(args.gpu_id)

    check_dir(args.output_path)
    train_path = os.path.join(args.output_path, "train")
    check_dir(train_path)
    ckpt_path = os.path.join(args.output_path, "checkpoints")
    check_dir(ckpt_path)

    train_list = read_list(args.list_path, args.data_path, mode="train")
    val_list = read_list(args.list_path, args.data_path, mode="val")

    agent = Agent(args)

    plot_driver = PlotDriver(output_path=train_path,
                             colors=['r', 'b', 'm', 'g', 'y', 'k', 'c'],
                             labels=['Loss', 'TranAng', 'TrainDis', 'TrainADI', 'ValAng', 'ValDis', 'ValADI'])
    best_adi = -90
    for epoch in range(1, 1+args.num_epoch):
        train_loss, train_angle, train_distance, train_adi = train_epoch(train_list, args, agent)
        val_angle, val_distance, val_adi = val_epoch(val_list, args, agent)

        plot_driver.update(train_loss, train_angle, train_distance, train_adi, val_angle, val_distance, val_adi)
        plot_driver.plot()

        if epoch % 1 == 0:
            torch.save(agent.eval_net.state_dict(), os.path.join(ckpt_path, "network_{}.pth.gz".format(epoch)))

        if best_adi < val_adi:
            best_adi = val_adi
            torch.save(agent.eval_net.state_dict(), os.path.join(ckpt_path, "best_adi.pth.gz"))

        info = [str(epoch).zfill(3), train_loss, train_angle, train_distance, train_adi, val_angle, val_distance, val_adi]
        print("Epoch {} "
              "|| Train-| Loss: {:.3f} Ang: {:.1f} Dis: {:.1f} ADI: {:+.1f} "
              "|| Val-| Ang: {:.1f} Dis: {:.1f} ADI: {:.1f}".format(*info))


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
