from models.Parameters import *
from itertools import count
from models.DQN.model import DQN
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
Games = ['CartPole', 'MountainCar-v0']


def draw_duration(durations):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    x = [i+1 for i in range(len(durations))]
    y = durations
    ax.plot(x, y)
    ax.set_xlabel("episode")
    ax.set_ylabel("duration")
    plt.show()


def main(args):
    writer = SummaryWriter(log_dir='logs', filename_suffix='pth')
    agent = DQN(args.game, args.gamma, args.batch_size,
                args.eps_start, args.eps_end, args.eps_decay,
                args.mem_size, args.device)
    episode_durations = []
    for i in range(args.num_episodes):
        agent.env.reset()
        last_screen = agent.env.get_screen()
        current_screen = agent.env.get_screen()
        state = current_screen - last_screen
        for t in count():
            action = agent.select_action(state)

            _, reward, done, _ = agent.env.env.step(action.item())
            reward = torch.tensor([reward], device=args.device)
            last_screen = current_screen
            current_screen = agent.env.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.optimize()

            if done:
                episode_durations.append(t+1)
                writer.add_scalar("duration", t+1, i)
                break
            else:
                print(f"Episode ## {i+1} ##, duration ## {t+1} ## survive")
        if i % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    agent.env.close()
    writer.close()
    return episode_durations


if __name__ == "__main__":
    durations = main(args)
    draw_duration(durations)
