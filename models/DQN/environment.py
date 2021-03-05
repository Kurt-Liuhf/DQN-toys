import gym
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
from models.Parameters import args
import matplotlib

Games = ['CartPole-v0', 'MountainCar-v0']

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

class Environment(object):

    def __init__(self, game_name):
        self.game_name = game_name
        self.env = gym.make('CartPole-v0').unwrapped
        self.reset()
        self.num_state = self.env.observation_space.shape[0]
        self.num_action = self.env.action_space.n

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)

    def get_screen(self, is_scale=True):
        # gym will return a 400*600*3 (height, width, channel) figure
        # but the conv2d needs the input shape to be (channel, height, width)
        # screen = self.env.render(mode='rgb_array').transpose(2, 0, 1)
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
        # don't know if need to scale the image
        if is_scale:
            resize = T.Compose([T.ToPILImage(),
                                T.Resize(40, interpolation=Image.CUBIC),
                                T.ToTensor])
            screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
            view_width = int(screen_width * 0.6)
            cart_location = self.get_cart_location(screen_width)
            if cart_location < view_width // 2:
                slice_range = slice(view_width)
            elif cart_location > (screen_width - view_width // 2):
                slice_range = slice(-view_width, None)
            else:
                slice_range = slice(cart_location - view_width // 2,
                                    cart_location + view_width // 2)
            screen = screen[:, :, slice_range]
        screen = np.ascontiguousarray(screen, dtype=np.float32)
        screen = torch.from_numpy(screen)

        return screen.unsqueeze(0).to(args.device)

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()


if __name__ == '__main__':
    env = Environment(Games[0])
    print(f"number of states: {env.num_state}")
    print(f"number of actions: {env.num_action}")




