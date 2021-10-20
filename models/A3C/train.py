import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from env import create_train_env
from model import ActorCritics
from optimizer import GlobalAdam
from process import local_train, local_test
import torch.multiprocessing as mp
import shutil


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("args parser")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_interval0", type=int, default=500)
    parser.add_argument("--max_actions", type=int, default=200)
    parser.add_argument("--log_path", type=str, default="log/a3c_super_mario")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_stage", type=bool, default=False)
    parser.add_argument("--use_gpu", type=bool, default=True)

    args = parser.parse_args()
    return args


def train(opt: argparse.ArgumentParser):
    torch.manual_seed(42)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
        os.makedirs(opt.log_path)
        if not os.path.isdir(opt.saved_path):
            os.makedirs(opt.saved_path)
    multi_processes = mp.get_context("spawn")
    env, num_states, num_actions = create_train_env(opt.world, opt.stage,
                                                    opt.action_type)
    global_model = ActorCritics(num_states, num_actions)
    if opt.use_gpu and torch.cuda.is_available():
        global_model.cuda()
    global_model.share_memory()
    if opt.load_from_stage:
        if opt.stage == 1:
            previous_worldd = opt.world - 1
            previous_stage = 4
        else:
            previous_world = opt.world
            previous_stage = opt.stage - 1
        file_ = f"{opt.saved_path}/a3c_super_mario_bros_{previous_world}_{previous_stage}"
        if os.path.isfile(file_):
            global_model.load_state_dict(file_)
    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    processes = []
    for pid in range(opt.num_processes):
        if pid == 0:
            process = multi_processes.Process(target=local_train,
                                              args=(pid,
                                                    opt,
                                                    global_model,
                                                    optimizer,
                                                    True))
        else:
            process = multi_processes.Process(target=local_train,
                                              args=(pid,
                                                    opt,
                                                    global_model,
                                                    optimizer))
        process.start()
        processes.append(process)
    process = multi_processes.Process(target=local_train,
                                      args=(opt.num_processes,
                                            opt,
                                            global_model,
                                            optimizer))
    process.start()
    processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = parse_args()
    train(opt)
