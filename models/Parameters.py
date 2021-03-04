import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
parser.add_argument("--mem_size", help="replay memory size", type=int, default=1e4)
parser.add_argument("--num_episode", help="number of episode", type=int, default=50)
parser.add_argument("--gamma", help="gamma of greedy search", type=float, default=0.9)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)
parser.add_argument('--device', help="device to train model on", type=str, default=None)
parser.add_argument('--game', help="name of the game", type=str, default="MountainCar-v0")


args = parser.parse_args()
if args.device is None:
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')