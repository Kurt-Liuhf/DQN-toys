from collections import namedtuple
import random


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.mem_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # loop the position to reuse
        self.position = (self.position + 1) % self.mem_size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
