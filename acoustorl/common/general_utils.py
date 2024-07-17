import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym

from acoustorl import *


def train_off_policy_agent(env, agent, replay_buffer, batch_size, minimal_size, total_timesteps, update_times, env_name, target_folder, seed):
    num_evaluate = 5
    num_timesteps = 0
    best_episode_return = 0
    return_list = []
    std_list = []

    while num_timesteps < total_timesteps:

        state, info = env.reset(seed=seed)
        terminated, truncated = False, False

        while not terminated and not truncated:
            if replay_buffer.memory_num < minimal_size:
                action = env.action_space.sample()
            else:
                action = agent.take_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            replay_buffer.store(state, action, reward, next_state, terminated)

            state = next_state

            if replay_buffer.memory_num > minimal_size:
                for _ in range(update_times):
                    agent.train(replay_buffer, batch_size)

            # Evaluate every 1% total timesteps, each evaluation reports the average reward over num_evaluate with no exploration noise.
            # The results are reported over 10 random seeds of the Gym simulator and the network initialization.
            if num_timesteps % (total_timesteps/100) == 0:
                average_episode_return, return_std = eval_policy(agent, env_name, num_evaluate)
                return_list.append(average_episode_return) 
                std_list.append(return_std)       
                if average_episode_return >= best_episode_return:
                    best_episode_return = average_episode_return
                    agent.save_experiment(target_folder, seed)

                percentage = num_timesteps/(total_timesteps/100)
                print("---------------------------------------")
                print(f"The No. {seed} th training has been finished: {percentage} %.\n")
                print("---------------------------------------")

            num_timesteps += 1
            if num_timesteps >= total_timesteps:
                break

    return return_list, std_list


# Runs policy/agent for X episodes and returns average reward and reward std
# Different X seeds are used for the eval environment
def eval_policy(agent, env_name, eval_episodes=10):
    eval_env = gym.make(env_name)

    avg_reward = np.zeros([eval_episodes])
    for i in range(eval_episodes):

        state, info = eval_env.reset(seed = i+100)
        terminated, truncated = False, False

        while not terminated and not truncated:
            action = agent.take_action(state, explore=False)

            next_state, reward, terminated, truncated, info = eval_env.step(action)

            state = next_state
            avg_reward[i] += reward

    average_reward = np.mean(avg_reward)
    reward_std = np.std(avg_reward, ddof=1)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {average_reward:.3f} +- {reward_std:.3f} ")
    print("---------------------------------------")

    return average_reward, reward_std


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.max_size = max_size
        self.ptr = 0
        self.memory_num = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = device

    def store(self, state, action, reward, next_state, terminated):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.memory_num = min(self.memory_num + 1, self.max_size)    

    def sample_batch(self, batch_size):
        ind = np.random.randint(0, self.memory_num, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
    

class ReplayBuffer_MADDPG():
    def __init__(self, num_agents, state_dim, action_dim, max_size=int(1e6), device=None):
        self.max_size = max_size
        self.ptr = 0
        self.memory_num = 0

        self.state = np.zeros((max_size, num_agents, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, num_agents, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, num_agents, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, num_agents), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = device

    def store(self, state, action, reward, next_state, terminated):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.memory_num = min(self.memory_num + 1, self.max_size)    

    def sample_batch(self, batch_size):
        ind = np.random.randint(0, self.memory_num, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
    

def algorithm_instantiation(args, kwargs):
    # Initialize agent
    if args.algorithm == "TD3":
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["policy_freq"] = args.policy_freq
        agent = TD3(**kwargs)
    elif args.algorithm == "TD3_per":
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["policy_freq"] = args.policy_freq
        kwargs["if_use_huber_loss"] = args.if_use_huber_loss
        agent = TD3_per(**kwargs)
    elif args.policy == "DDPG":
        agent = DDPG(**kwargs)
    elif args.policy == "DDPG_per":
        kwargs["if_use_huber_loss"] = args.if_use_huber_loss
        agent = DDPG_per(**kwargs)

    return agent 


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            agent.save(i)
    return return_list