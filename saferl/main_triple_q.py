import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac_triple_q import SAC_QQQ
import os
import pandas as pd
from replay_memory import ReplayMemory
import json




def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while (eval_env.unwrapped.state[0] > 0.3 or eval_env.unwrapped.state[0] < -0.3):
            state = eval_env.reset()
        while not done:
            action = agent.select_action(np.array(state),evaluate=True)
            state, reward, done, _ = eval_env.step(action)
            angle = abs(np.arctan2(state[1], state[0]))
            cost = 1 if angle >= 1 else 0
            avg_reward += reward
            avg_cost += cost
    avg_reward /= eval_episodes
    avg_cost /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: reward: {avg_reward:.3f} cost: {avg_cost:.3f}")
    print("---------------------------------------")
    return avg_reward, avg_cost





parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Pendulum-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--eta', type=float, default=2000, metavar='G',)
parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=80000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=200, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument("--eval_freq", default=2000, type=int, help="evaluation frequency")
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

file_name = f"SAC_{args.env_name}_{args.seed}"
print("---------------------------------------")
print(f"Env: {args.env_name}, Seed: {args.seed}")
print("---------------------------------------")


# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC_QQQ(env.observation_space.shape[0], env.action_space, args)


# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

#evaluations = [eval_policy(agent, args.env_name, args.seed)]
evaluations = []

variant = dict(
    algorithm='SAC_QQQ',
    env=args.env_name,
)

if not os.path.exists(f"./qqq/{args.env_name}/SAC_qqq/seed{args.seed}"):
    os.makedirs(f'./qqq/{args.env_name}/SAC_qqq/seed{args.seed}')

with open(f'./qqq/{args.env_name}/SAC_qqq/seed{int(args.seed)}/variant.json', 'w') as outfile:
    json.dump(variant,outfile)

c_list = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_cost = 0
    max_c = 0
    episode_steps = 0
    done = False
    state = env.reset()
    # Ensure that starting position is in "safe" region
    while (env.unwrapped.state[0] > 0.3 or env.unwrapped.state[0] < -0.3):
        state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, critic_1_loss_c, critic_2_loss_c, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                updates += 1

        next_state, reward, done, _ = env.step(action) # Step


        angle = abs(np.arctan2(next_state[1], next_state[0]))
        max_c = angle if angle > max_c else max_c
        cost = 1 if angle > 0.5 else 0

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        episode_cost += cost


        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push_q(state, action, reward, 1 - cost, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break
    c_list.append(200 - episode_cost)
    if int(i_episode) % 5 == 0:
        queue_ = max(agent.queue + 195 - np.mean(c_list), 0)
        agent.queue = queue_
        c_list = []

    rrr = episode_reward
    ccc = max_c
    evaluations.append([rrr,ccc])
    data = np.array(evaluations)
    df = pd.DataFrame(data=data,columns=["Average Returns","Max Angle"]).reset_index()
    df['env'] = args.env_name
    df['algorithm_name'] = 'SAC_qqq'
    df.to_csv(f'./qqq/{args.env_name}/SAC_qqq/seed{args.seed}/progress.csv', index = False)

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2), round(episode_cost, 2)))
    print(f"virtual: {agent.queue}, max_c: {max_c}")
    if int(i_episode) > 100:
        break


env.close()
