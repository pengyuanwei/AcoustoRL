import gymnasium as gym
import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


# Generate a total learning curve based on 5 sub learning curves: the bold line in the shaded area consists of 
# the mean of the five average rewards at each timestep, and the shaded contour line consists of the quartiles of the mean.
if __name__ == "__main__":
    # Define the save path
    target_folder = "../../../test_results/figs"

    # Check the target folder, if not, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    save_name = target_folder + '/learning_curve_1.svg'

    # Create a subplot
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    num_experiment = 5
    env_name = ['Hopper-v3', 'HalfCheetah-v2', 'Walker2d-v4']
    algorithm = ['TD3', 'TD3_multi_update_times', 'TD3_multi_update_times_v1']




    ####################################################################################################
    target_folder = "../../../test_results/%s_%s"%(env_name[0], algorithm[0])

    # 读取
    total_return_array = np.zeros([num_experiment,100])
    for i in range(num_experiment):
        file_name = f"return_list{i}.npy"
        target_file = os.path.join(target_folder, file_name)
        return_list = np.load(target_file)
        #return_list = return_list[-90:]
        total_return_array[i] = return_list

    # Calculate the mean
    avg_return = np.mean(total_return_array, axis=0)

    # 计算第 25 和第 75 百分位数 (用于阴影区域)
    Q1 = np.percentile(total_return_array, 25, axis=0)
    Q3 = np.percentile(total_return_array, 75, axis=0)

    # 绘制阴影区域
    axs[0].fill_between(range(avg_return.shape[0]), Q1, Q3, color=(0.0, 0.0, 1.0, 0.1), alpha=0.2)
    # 绘制平均值曲线
    line0, = axs[0].plot(avg_return, label=algorithm[0], color=(0.0, 0.0, 1.0, 1.0), linewidth=2)

    ####################################################################################################
    target_folder = "../../../test_results/%s_%s"%(env_name[0], algorithm[1])

    # 读取
    total_return_array = np.zeros([num_experiment,100])
    for i in range(num_experiment):
        file_name = f"return_list{i}.npy"
        target_file = os.path.join(target_folder, file_name)
        return_list = np.load(target_file)
        #return_list = return_list[-90:]
        total_return_array[i] = return_list

    # Calculate the mean
    avg_return = np.mean(total_return_array, axis=0)

    # 计算第 25 和第 75 百分位数 (用于阴影区域)
    Q1 = np.percentile(total_return_array, 25, axis=0)
    Q3 = np.percentile(total_return_array, 75, axis=0)

    # 绘制阴影区域
    axs[0].fill_between(range(avg_return.shape[0]), Q1, Q3, color=(0.0, 1.0, 0.0, 0.1), alpha=0.2)
    # 绘制平均值曲线
    line1, = axs[0].plot(avg_return, label=algorithm[1], color=(0.0, 1.0, 0.0, 1.0), linewidth=2)

    ####################################################################################################
    target_folder = "../../../test_results/%s_%s"%(env_name[0], algorithm[2])

    # 读取
    total_return_array = np.zeros([num_experiment,100])
    for i in range(num_experiment):
        file_name = f"return_list{i}.npy"
        target_file = os.path.join(target_folder, file_name)
        return_list = np.load(target_file)
        #return_list = return_list[-90:]
        total_return_array[i] = return_list

    # Calculate the mean
    avg_return = np.mean(total_return_array, axis=0)

    # 计算第 25 和第 75 百分位数 (用于阴影区域)
    Q1 = np.percentile(total_return_array, 25, axis=0)
    Q3 = np.percentile(total_return_array, 75, axis=0)

    # 绘制阴影区域
    axs[0].fill_between(range(avg_return.shape[0]), Q1, Q3, color=(1.0, 0.0, 0.0, 0.1), alpha=0.2)
    # 绘制平均值曲线
    line2, = axs[0].plot(avg_return, label=algorithm[2], color=(1.0, 0.0, 0.0, 1.0), linewidth=2)




    ####################################################################################################
    target_folder = "../../../test_results/%s_%s"%(env_name[1], algorithm[0])

    # 读取
    total_return_array = np.zeros([num_experiment,100])
    for i in range(num_experiment):
        file_name = f"return_list{i}.npy"
        target_file = os.path.join(target_folder, file_name)
        return_list = np.load(target_file)
        #return_list = return_list[-90:]
        total_return_array[i] = return_list

    # Calculate the mean
    avg_return = np.mean(total_return_array, axis=0)

    # 计算第 25 和第 75 百分位数 (用于阴影区域)
    Q1 = np.percentile(total_return_array, 25, axis=0)
    Q3 = np.percentile(total_return_array, 75, axis=0)

    # 绘制阴影区域
    axs[1].fill_between(range(avg_return.shape[0]), Q1, Q3, color=(0.0, 0.0, 1.0, 0.1), alpha=0.2)
    # 绘制平均值曲线
    axs[1].plot(avg_return, label=algorithm[0], color=(0.0, 0.0, 1.0, 1.0), linewidth=2)

    ####################################################################################################
    target_folder = "../../../test_results/%s_%s"%(env_name[1], algorithm[1])

    # 读取
    total_return_array = np.zeros([num_experiment,100])
    for i in range(num_experiment):
        file_name = f"return_list{i}.npy"
        target_file = os.path.join(target_folder, file_name)
        return_list = np.load(target_file)
        #return_list = return_list[-90:]
        total_return_array[i] = return_list

    # Calculate the mean
    avg_return = np.mean(total_return_array, axis=0)

    # 计算第 25 和第 75 百分位数 (用于阴影区域)
    Q1 = np.percentile(total_return_array, 25, axis=0)
    Q3 = np.percentile(total_return_array, 75, axis=0)

    # 绘制阴影区域
    axs[1].fill_between(range(avg_return.shape[0]), Q1, Q3, color=(0.0, 1.0, 0.0, 0.1), alpha=0.2)
    # 绘制平均值曲线
    axs[1].plot(avg_return, label=algorithm[1], color=(0.0, 1.0, 0.0, 1.0), linewidth=2)

    ####################################################################################################
    target_folder = "../../../test_results/%s_%s"%(env_name[1], algorithm[2])

    # 读取
    total_return_array = np.zeros([num_experiment,100])
    for i in range(num_experiment):
        file_name = f"return_list{i}.npy"
        target_file = os.path.join(target_folder, file_name)
        return_list = np.load(target_file)
        #return_list = return_list[-90:]
        total_return_array[i] = return_list

    # Calculate the mean
    avg_return = np.mean(total_return_array, axis=0)

    # 计算第 25 和第 75 百分位数 (用于阴影区域)
    Q1 = np.percentile(total_return_array, 25, axis=0)
    Q3 = np.percentile(total_return_array, 75, axis=0)

    # 绘制阴影区域
    axs[1].fill_between(range(avg_return.shape[0]), Q1, Q3, color=(1.0, 0.0, 0.0, 0.1), alpha=0.2)
    # 绘制平均值曲线
    axs[1].plot(avg_return, label=algorithm[2], color=(1.0, 0.0, 0.0, 1.0), linewidth=2)




    ####################################################################################################
    target_folder = "../../../test_results/%s_%s"%(env_name[2], algorithm[0])

    # 读取
    total_return_array = np.zeros([num_experiment,100])
    for i in range(num_experiment):
        file_name = f"return_list{i}.npy"
        target_file = os.path.join(target_folder, file_name)
        return_list = np.load(target_file)
        #return_list = return_list[-90:]
        total_return_array[i] = return_list

    # Calculate the mean
    avg_return = np.mean(total_return_array, axis=0)

    # 计算第 25 和第 75 百分位数 (用于阴影区域)
    Q1 = np.percentile(total_return_array, 25, axis=0)
    Q3 = np.percentile(total_return_array, 75, axis=0)

    # 绘制阴影区域
    axs[2].fill_between(range(avg_return.shape[0]), Q1, Q3, color=(0.0, 0.0, 1.0, 0.1), alpha=0.2)
    # 绘制平均值曲线
    axs[2].plot(avg_return, label=algorithm[0], color=(0.0, 0.0, 1.0, 1.0), linewidth=2)

    ####################################################################################################
    target_folder = "../../../test_results/%s_%s"%(env_name[2], algorithm[1])

    # 读取
    total_return_array = np.zeros([num_experiment,100])
    for i in range(num_experiment):
        file_name = f"return_list{i}.npy"
        target_file = os.path.join(target_folder, file_name)
        return_list = np.load(target_file)
        #return_list = return_list[-90:]
        total_return_array[i] = return_list

    # Calculate the mean
    avg_return = np.mean(total_return_array, axis=0)

    # 计算第 25 和第 75 百分位数 (用于阴影区域)
    Q1 = np.percentile(total_return_array, 25, axis=0)
    Q3 = np.percentile(total_return_array, 75, axis=0)

    # 绘制阴影区域
    axs[2].fill_between(range(avg_return.shape[0]), Q1, Q3, color=(0.0, 1.0, 0.0, 0.1), alpha=0.2)
    # 绘制平均值曲线
    axs[2].plot(avg_return, label=algorithm[1], color=(0.0, 1.0, 0.0, 1.0), linewidth=2)

    ####################################################################################################
    target_folder = "../../../test_results/%s_%s"%(env_name[2], algorithm[2])

    # 读取
    total_return_array = np.zeros([num_experiment,100])
    for i in range(num_experiment):
        file_name = f"return_list{i}.npy"
        target_file = os.path.join(target_folder, file_name)
        return_list = np.load(target_file)
        #return_list = return_list[-90:]
        total_return_array[i] = return_list

    # Calculate the mean
    avg_return = np.mean(total_return_array, axis=0)

    # 计算第 25 和第 75 百分位数 (用于阴影区域)
    Q1 = np.percentile(total_return_array, 25, axis=0)
    Q3 = np.percentile(total_return_array, 75, axis=0)

    # 绘制阴影区域
    axs[2].fill_between(range(avg_return.shape[0]), Q1, Q3, color=(1.0, 0.0, 0.0, 0.1), alpha=0.2)
    # 绘制平均值曲线
    axs[2].plot(avg_return, label=algorithm[2], color=(1.0, 0.0, 0.0, 1.0), linewidth=2)




    # 添加标签和图例
    axs[0].set_xlabel('Timesteps (1e4)')
    axs[0].set_ylabel('Returns')
    axs[0].set_title(env_name[0])

    axs[1].set_xlabel('Timesteps (1e4)')
    axs[1].set_ylabel('Returns')
    axs[1].set_title(env_name[1])

    axs[2].set_xlabel('Timesteps (1e4)')
    axs[2].set_ylabel('Returns')
    axs[2].set_title(env_name[2])

    fig.legend(handles=[line0, line1, line2], labels=algorithm, loc='upper center', ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_name, format='svg')    
    
    # 显示图形
    plt.show()

