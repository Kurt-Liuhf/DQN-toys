import gym
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter


class Actor(nn.Module):
    
    def __init__(self, num_state, num_action, hidden_state):
        super(Actor, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.hidden_state = hidden_state
        self.fc1 = nn.Linear(self.num_state, self.hidden_state)
        self.action_head = nn.Linear(self.hidden_state, self.num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.action_head(x), dim=1)

        return action_probs


class Critics(nn.Module):
    
    def __init__(self, num_state, hidden_state):
        super(Critics, self).__init__()
        self.num_state = num_state
        self.hidden_state = hidden_state
        self.fc1 = nn.Linear(self.num_state, self.hidden_state)
        self.state_value = nn.Linear(self.hidden_state, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_value = self.state_value(x)

        return state_value


class PPO(object):
    
    def __init__(self,
                 num_state,
                 num_action,
                 hidden_state=128,
                 actor_lr=1e-3,
                 critics_lr=3e-4,
                 batch_size=64,
                 gamma=0.3,
                 clip_param=0.2,
                 max_grad_norm=0.5,
                 update_time=10,
                 mem_size=1e4):
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.update_time = update_time
        self.mem_size = mem_size

        self.memory = []
        self.cnt = 0
        self.training_step = 0
        self.actor = Actor(num_state, num_action, hidden_state)
        self.critics = Critics(num_state, hidden_state)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), actor_lr)
        self.critics_optimizer = optim.Adam(self.critics.parameters(), critics_lr)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_state_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critics(state)
        return state.item()

    def store_transitions(self, transition):
        self.memory.append(transition)
        self.cnt += 1

    def update(self, num_episode):
        state = torch.tensor([t.state for t in self.memory], dtype=torch.float)
        action = torch.tensor([t.action for t  in self.memory], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.memory]
        last_action_log_prob = torch.tensor([t.a_log_prob for t in self.memory], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        for i in range(self.update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))),
                                      self.batch_size,
                                      False):
                if self.training_step % 100 == 0:
                    print(f'Episode #{num_episode}#, train #{self.training_step}# times.')
                Gt_index = Gt[index].view(-1, 1)
                state_values = self.critics(state[index])
                tmp_adv = Gt_index - state_values
                advantage = tmp_adv.detach()

                action_prob = self.actor(state[index]).gather(1, action[index])
                ratio = action_prob / last_action_log_prob[index]
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                action_loss = -1 * torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, state_values)
                self.critics_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm(self.critics.parameters(), self.max_grad_norm)
                self.critics_optimizer.step()

                self.training_step += 1

        del self.memory[:]


def main():
    pass


if __name__=="__main__":
    pass