import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.Parameters import args
from models.DQN.environment import Environment, Games
import random
import numpy as np
from utils.ReplayMemory import *


class Net(nn.Module):

    def __init__(self, height, width, outputs):
        """
        :param height: height of the raw image (observation)
        :param width: width of the raw image (observation)
        :param outputs: number of outputs (actions)
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # calculate the output size after the Conv operations
        def conv2_size_out(original_size, kernal_size, stride):
            return (original_size - kernal_size) // stride + 1

        height_after_conv = conv2_size_out(conv2_size_out(conv2_size_out(height)))
        width_after_conv = conv2_size_out(conv2_size_out(conv2_size_out(width)))
        size_after_conv = height_after_conv * width_after_conv * 32  # 32 conv kernel
        self.fc = nn.Linear(size_after_conv, outputs)

    def forward(self, x):
        # x: (batch_size, in_channel, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # flatten the conv outputs -> (batch_size, size_after_conv)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQN(object):

    def __init__(self, game_name, gamma, batch_size,
                 eps_start, eps_end, eps_decay,
                 mem_size, device):
        if batch_size > mem_size:
            print("Error: the training crushes due to batch size smaller than memory size.")
            return
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.env = Environment(game_name)
        self.step_done = 0
        self.device = device
        self.memory = ReplayMemory(mem_size)
        # define the policy net and target net
        _, _, height, width = self.env.get_screen().shape
        self.policy_net = Net(height, width, self.env.num_action).to(self.device)
        self.target_net = Net(height, width, self.env.num_action).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end)\
                        * np.exp(-1 * self.step_done / self.eps_decay)
        self.step_done += 1
        # decide whether to exploitation or exploration
        if sample > eps_threshold:
            with torch.no_grad():
                # return the action with the largest expected reward
                # similar to classification task but not the same
                # both tasks use the scoring mechanism to achieve their goals
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.env.num_action)]],
                                device=self.device, dtype=torch.long)

    def optimize(self):
        # see https://stackoverflow.com/a/19343/3343043
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # creat masks
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # use the policy_net as the behavior network
        # use the target_net as the Q-values fitting network
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # compute the loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def plot_duration(self):
        pass
