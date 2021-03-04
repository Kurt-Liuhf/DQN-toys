import gym
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch


Games = ['MountainCar-v0', 'CartPole']


class Environment(object):

    def __init__(self, game_name):
        self.game_name = game_name
        self.env = gym.make(game_name).unwrapped
        self.num_state = env.observation_space.shape[10]
        self.num_action = env.action_space.n

    def get_cart_location(self, screen_width):
        world_width = env.x_thredshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.0)

    def get_screen(self, is_scale=True):
        # gym will return a 400*600*3 (height, width, channel) figure
        # but the conv2d needs the input shape to be (channel, height, width)
        screen = env.render(mode='rgb_array').transpose(2, 0, 1)
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

        return screen.unsqueeze(0)


if __name__ == '__main__':
    env, a, b = create_env(Games[0])
    print(a)
    print(b)
    print(env.action_space)




