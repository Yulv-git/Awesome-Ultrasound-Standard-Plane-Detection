import random

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg13_bn
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = vgg13_bn(pretrained=False).features
        self.linear1 = nn.Linear(512*7*7, 512)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)

        self.fc1_adv = nn.Linear(in_features=512, out_features=128)
        self.fc1_val = nn.Linear(in_features=512, out_features=128)

        self.fc2_adv = nn.Linear(in_features=128, out_features=8)
        self.fc2_val = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.drop(self.relu(self.linear1(x)))

        adv = self.drop(self.relu(self.fc1_adv(x)))
        val = self.drop(self.relu(self.fc1_val(x)))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), 8)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), 8)

        return x


class Agent(object):
    """
    we define the Double-DQN agent
    """
    def __init__(self, args):
        # define the eval net and target net.
        self.eval_net = Net().cuda()
        self.target_net = Net().cuda()

        # define the batch size
        self.batch_size = args.batch_size
        # define the target net weight step counter
        self.target_step_counter = args.target_step_counter
        # define the replay buffer
        self.replay_buffer = NaivePrioritizedBuffer(args.memory_capacity)
        # define the optimizer
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # define the loss function
        self.loss_func = nn.MSELoss()
        # define the learning step
        self.learning_step = 0

        # define the epsilon of the greedy algorithm
        self.epsilon = args.epsilon
        # define the reward decay
        self.gamma = args.gamma
        self.frame_idx = 0

    def select_action(self, x, return_q=False, random_action=True):
        self.eval_net.eval()
        x = torch.from_numpy(x[np.newaxis, :, :, :]).cuda()
        if random_action:
            if random.random() < self.epsilon:
                action = self.eval_net.forward(x)
                action_value_arr = action.data.cpu().numpy()
            else:
                # select the random action
                action_value_arr = np.array([random.random() for _ in range(8)], dtype=np.float32)
        else:
            action = self.eval_net.forward(x)
            action_value_arr = action.data.cpu().numpy()

        action_value_arr = np.squeeze(action_value_arr)
        action = action_value_arr.argmax(0)
        if return_q:
            return action, action_value_arr
        else:
            return action

    def learn(self, beta):
        self.eval_net.train()
        self.target_net.eval()
        if self.learning_step % self.target_step_counter == 0:
            # update the target weight after target step counter
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step += 1

        if len(self.replay_buffer) < self.batch_size:
            return 0

        if self.learning_step % 10000 == 0 and self.epsilon < 0.95:
            # increase the epsilon to reduce the ability of explore
            self.epsilon *= 1.01

        # sample the state from the buffer.
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(self.batch_size, beta)

        # create variable
        state = Variable(torch.from_numpy(state)).cuda()
        next_state = Variable(torch.from_numpy(next_state)).cuda()
        reward = Variable(torch.from_numpy(reward)).cuda()
        action = Variable(torch.from_numpy(action[:, np.newaxis])).cuda()
        weights = Variable(torch.from_numpy(weights)).cuda()

        # get the current q value
        q_eval = self.eval_net(state)
        q_eval = q_eval.gather(1, action).squeeze()

        # get the target q value and forbid the gradient
        q_next = self.target_net(next_state).detach()
        q_target = (q_next.max(1)[0] * self.gamma) + reward

        # calculate the loss function
        loss = self.loss_func(q_eval, q_target) * weights
        prios = loss + 1e-5

        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()

        # update the replay buffer weights
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        return loss.item()


class NaivePrioritizedBuffer(object):
    """
    This is the naive prioritized Buffer that borrowed from the Github (forgot the source....)
    """
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = zip(*samples)
        batch = [data for data in batch]

        states = np.concatenate(batch[0], axis=0)
        actions = np.array(batch[1], dtype=np.int64)
        rewards = np.array(batch[2], dtype=np.float32)
        next_states = np.concatenate(batch[3], axis=0)
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def beta_by_frame(frame_idx):
    beta_start = 0.4
    beta_frames = 2000
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
