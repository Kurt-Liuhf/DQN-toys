import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.Parameters import args


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


class DQN():

    def __init__(self):
        pass

    def select_action(self):
        pass

    def plot_duration(self):
        pass
