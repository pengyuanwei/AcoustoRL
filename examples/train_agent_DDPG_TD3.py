import os
import torch
import random
import numpy as np
import gymnasium as gym

from argparse import ArgumentParser

from acoustorl import *
from acoustorl.common import per
from acoustorl.common.general_utils import *


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

    return agent 


# Train models separately using 5 different random seeds.
# During each training, every time evaluation uses 5 random seeds different from the training random seed.
# After each evaluation, calculate the average reward obtained by the model and save it.
if __name__ == "__main__":
    parser = ArgumentParser(description='ArgumentParser for AcoustoRL')
    parser.add_argument(        "--algorithm", default="TD3", type=str, help='Algorithm name')
    parser.add_argument(              "--env", default="HalfCheetah-v2", type=str, help='Environment name')       
    parser.add_argument(       "--hidden_dim", default=256, type=int, help='The number of neurons in the hidden layer')     
    parser.add_argument("--exploration_noise", default=0.1, type=float, help='Exploration noise when select action')             
    parser.add_argument(         "--discount", default=0.99, type=float, help='Discount factor')    
    parser.add_argument(              "--tau", default=0.005, type=float, help='Soft update rate of target network')        
    parser.add_argument(         "--actor_lr", default=3e-4, type=float, help='The learning rate of actor network')        
    parser.add_argument(        "--critic_lr", default=3e-4, type=float, help='The learning rate of critic network')        
    parser.add_argument(     "--policy_noise", default=0.2, type=float, help='Noise added to target policy during critic update')             
    parser.add_argument(       "--noise_clip", default=0.5, type=float, help='Range to clip target policy noise')               
    parser.add_argument(      "--policy_freq", default=1, type=int, help='Frequency of delayed policy updates')      
    parser.add_argument(     "--update_times", default=1, type=int, help='The number of gradient descent steps performed per update')      
    parser.add_argument("--if_use_huber_loss", default=False, type=bool, help='If use Huber loss')      
    parser.add_argument(  "--total_timesteps", default=1e6, type=int, help='Max time steps to run environment')   
    parser.add_argument(      "--buffer_size", default=1e6, type=int, help='Buffer size')       
    parser.add_argument(     "--minimal_size", default=25e3, type=int, help='Time steps initial random policy is used')
    parser.add_argument(       "--batch_size", default=256, type=int, help='Batch size for both actor and critic')     
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("---------------------------------------")
    print(f"Algorithm: {args.algorithm}, Env: {args.env}, Device: {device}")
    print("---------------------------------------")

    # Define the save path
    target_folder = "../../test_results/%s_%s"%(args.env, args.algorithm)
    # Check the target folder, if not, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for i in range(5):
        env = gym.make(args.env)

        # Set seeds
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        min_action = float(env.action_space.low[0])
        max_action = float(env.action_space.high[0])  # 动作最大值

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "min_action": min_action,
            "max_action": max_action,
            "hidden_dim": args.hidden_dim,
            "exploration_noise": args.exploration_noise,
            "discount": args.discount,
            "tau": args.tau,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "device": device
        }

        agent = algorithm_instantiation(args, kwargs)

        if args.algorithm == "TD3_per":
            replay_buffer = per(
                state_dim = state_dim,
                action_dim = action_dim,
                max_size = args.buffer_size,
                device = device
            )
        else:
            replay_buffer = ReplayBuffer(
                state_dim = state_dim,
                action_dim = action_dim,
                max_size = args.buffer_size,
                device = device
            )

        return_list, std_list = train_off_policy_agent_experiment_independent_buffer(
            env, 
            agent, 
            replay_buffer, 
            args.batch_size, 
            args.minimal_size, 
            args.total_timesteps, 
            args.env, 
            target_folder, 
            i
        )

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