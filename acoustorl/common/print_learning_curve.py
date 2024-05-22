import gymnasium as gym
import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from TD3_v1 import TD3
import os, sys

import planning_gym
from planning_gym.wrappers.discrete_action import DiscreteActions
from planning_gym.utils import general_utils


if __name__ == "__main__":
    total_timesteps = 1000000
    minimal_buffer_size = 100000
    eval_interval = 5000
    env_name = "Hopper-v3"
    num_experiment = 10


    # 读取
    total_return_array = np.zeros([num_experiment,180])

    for i in range(num_experiment):
        return_list = np.load('TD3_%s_data_v1/return_list%d.npy'%(env_name, i))
        return_list = return_list[-180:]
        total_return_array[i] = return_list

    sum_return_array = total_return_array.sum(axis=0)
    avg_return = sum_return_array/num_experiment

    transpose_array = total_return_array.T

    return_std = np.zeros(len(transpose_array))
    for i in range(len(transpose_array)):
        return_std[i] = np.std(transpose_array[i], ddof=1)

    print(avg_return.shape)
    print(return_std.shape)

    avg_return = avg_return.tolist()
    return_std = return_std.tolist()

    upper_bound = [avg_return[i] + return_std[i]/2 for i in range(len(avg_return))]
    lower_bound = [avg_return[i] - return_std[i]/2 for i in range(len(avg_return))]
        
    episodes_list = list(range(len(avg_return)))
    episodes_list = [x/199 for x in episodes_list]

    plt.fill_between(episodes_list, upper_bound, lower_bound, color=(1.0, 0.0, 0.0, 0.1))
    plt.plot(episodes_list, avg_return, color=(1.0, 0.0, 0.0, 1.0), label='Policy update delay = 1')
    plt.xlabel('Time steps (1e6)')
    plt.ylabel('Average Return')
    plt.title('TD3 on {}'.format(env_name))

    # 读取
    total_return_array = np.zeros([num_experiment,180])

    for i in range(num_experiment):
        return_list = np.load('TD3_%s_data_v0/return_list%d.npy'%(env_name, i))
        return_list = return_list[-180:]
        total_return_array[i] = return_list

    sum_return_array = total_return_array.sum(axis=0)
    avg_return = sum_return_array/num_experiment

    transpose_array = total_return_array.T

    return_std = np.zeros(len(transpose_array))
    for i in range(len(transpose_array)):
        return_std[i] = np.std(transpose_array[i], ddof=1)

    print(avg_return.shape)
    print(return_std.shape)

    avg_return = avg_return.tolist()
    return_std = return_std.tolist()

    upper_bound = [avg_return[i] + return_std[i]/4 for i in range(len(avg_return))]
    lower_bound = [avg_return[i] - return_std[i]/4 for i in range(len(avg_return))]
        
    episodes_list = list(range(len(avg_return)))
    episodes_list = [x/199 for x in episodes_list]

    plt.fill_between(episodes_list, upper_bound, lower_bound, color=(0.0, 0.0, 1.0, 0.1))
    plt.plot(episodes_list, avg_return, color=(0.0, 0.0, 1.0, 1.0), label='Policy update delay = 2')


    # 读取
    total_return_array = np.zeros([num_experiment,180])

    for i in range(num_experiment):
        return_list = np.load('TD3_%s_data/return_list%d.npy'%(env_name, i))
        return_list = return_list[-180:]
        total_return_array[i] = return_list

    sum_return_array = total_return_array.sum(axis=0)
    avg_return = sum_return_array/num_experiment

    transpose_array = total_return_array.T

    return_std = np.zeros(len(transpose_array))
    for i in range(len(transpose_array)):
        return_std[i] = np.std(transpose_array[i], ddof=1)

    print(avg_return.shape)
    print(return_std.shape)

    avg_return = avg_return.tolist()
    return_std = return_std.tolist()

    upper_bound = [avg_return[i] + return_std[i]/4 for i in range(len(avg_return))]
    lower_bound = [avg_return[i] - return_std[i]/4 for i in range(len(avg_return))]
        
    episodes_list = list(range(len(avg_return)))
    episodes_list = [x/199 for x in episodes_list]

    plt.fill_between(episodes_list, upper_bound, lower_bound, color=(0.0, 1.0, 0.0, 0.1))
    plt.plot(episodes_list, avg_return, color=(0.0, 1.0, 0.0, 1.0), label='Policy update delay = 3')
    plt.legend(loc='lower right')


    plt.xlabel('Time steps (1e6)')
    plt.ylabel('Average Return')
    plt.title('TD3 on {}'.format(env_name))

    plt.show()
    plt.savefig('learning_curve.png')
    #plt.clf()