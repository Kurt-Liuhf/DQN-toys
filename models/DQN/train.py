from models.Parameters import *
from itertools import count
from models.DQN.model import DQN


Games = ['CartPole', 'MountainCar-v0']


def main(args):
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
                break
            else:
                print(f"Episode ## {i+1} ##, duration ## {t+1} ## survive")
        if i % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    agent.env.close()


if __name__ == "__main__":
    main(args)