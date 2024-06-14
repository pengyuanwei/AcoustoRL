import gymnasium as gym
import random
import numpy as np
import collections
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from acoustorl.DDPG import DDPG
from acoustorl.common import general_utils


# Train models separately using 5 different random seeds.
# During each training, every time evaluation uses 5 random seeds different from the training random seed.
# After each evaluation, calculate the average reward obtained by the model and save it.
if __name__ == "__main__":
    env_name = 'Pendulum-v1'
    algorithm = 'DDPG'

    # Define hyperparameters
    total_timesteps = 50000
    discount = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 50000
    minimal_size = 1000
    batch_size = 256
    exploration_noise = 0.1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("The device is:", device)
    print("Using", algorithm, "on the environment", env_name)

    # Define the save path
    target_folder = "../../test_results/%s_%s"%(env_name, algorithm)

    # Check the target folder, if not, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for i in range(5):
        env = gym.make(env_name)

        # Set seeds
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        min_action = float(env.action_space.low[0])
        max_action = float(env.action_space.high[0])  # 动作最大值

        replay_buffer = general_utils.ReplayBuffer(state_dim = state_dim,
                                                   action_dim = action_dim,
                                                   max_size = buffer_size,
                                                   device = device)
        
        agent = DDPG(
            state_dim = state_dim, 
            action_dim = action_dim, 
            min_action = min_action,
            max_action = max_action, 
            exploration_noise = exploration_noise, 
            discount = discount, 
            tau = tau, 
            device = device
        )

        return_list, std_list = general_utils.train_off_policy_agent_experiment_independent_buffer(env, agent, replay_buffer, batch_size, minimal_size, total_timesteps, env_name, target_folder, i)

        # Define the the path of target file
        file_name1 = f"return_list{i}.npy"
        target_file1 = os.path.join(target_folder, file_name1)
        file_name2 = f"std_list{i}.npy"
        target_file2 = os.path.join(target_folder, file_name2)

        # Save the return and std
        return_list=np.array(return_list)
        np.save(target_file1, return_list)   # 保存为.npy格式
        std_list=np.array(std_list)
        np.save(target_file2, std_list)   # 保存为.npy格式