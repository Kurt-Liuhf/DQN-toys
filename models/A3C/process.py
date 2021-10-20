import torch
from env import create_train_env
from model import ActorCritics
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
import logging


def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(42+index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    state = torch.from_numpy(env.reset())
    local_model = ActorCritics(num_states, num_actions)
    if opt.use_gpu and torch.cuda.is_available():
        local_model = local_model.cuda()
        state = state.cuda()
    local_model.train()
    done = True
    cur_step = 0
    cur_episode = 0
    while True:
        if save:
            if cur_episode % opt.save_interval == 0 and cur_episode > 0:
                torch.save(global_model.state_dict(),
                           f"{opt.saved_path}/a3c_super_mario_bros_{opt.world}_{opt.stage}")
                print(f"Process {index}. Episode {cur_episode}")
            cur_episode += 1
            local_model.load_state_dict(global_model.state_dict())
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()
            if opt.use_gpu and torch.cuda.is_available():
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []
        # predict the action and react with the environment
        for _ in range(opt.num_local_steps):
            cur_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            # get the next action from sampling
            m = Categorical(policy)
            action = m.sample().item()
            # react
            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state)
            if opt.use_gpu and torch.cuda.is_available():
                state = state.cuda()
            # finishing of the episode
            if cur_step > opt.num_global_steps:
                done = True

            if done:
                cur_step = 0
                state = torch.from_numpy(env.reset())
                if opt.use_gpu and torch.cuda.is_available():
                    state = state.cuda()

            # aggregate the info
            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        # calculate the `R' and calculate loss according age
        R = torch.zeros((1, 1), dtype=torch.float)
        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu and torch.cuda.is_available():
            R = R.cuda()
            gae = gae.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        actor_loss, critic_loss, entropy_loss = 0, 0, 0
        next_value = R
        for value, log_policy, reward, entropy in list(values, log_policies, rewards, entropies)[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
        # backward
        total_loss  = critic_loss - actor_loss - opt.beta * entropy_loss
        writer.add_scalar(f"Train_{index}/Loss", total_loss, cur_episode)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if cur_episode == int(opt.num_global_steps / opt.num_local_steps):
            print(f"Training process {index} terminated")
            if save:
                end_time = timeit.default_timer()
                print(f"The code runs for {end_time -start_time} s")
            return


def local_test(index, opt, global_model):
    torch.manual_seed(42+index)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritics(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    cur_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        cur_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        # inference
        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        env.render()
        actions.append(action)
        if cur_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            cur_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)

