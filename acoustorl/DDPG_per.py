import os
import copy
import torch
import numpy as np
import torch.nn as nn

from acoustorl.actor_critic import Actor, Critic


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# Reference: https://github.com/sfujim/TD3/blob/master/OurDDPG.py


class DDPG_per():
	def __init__(
		self, 
		state_dim, 
		action_dim, 
		min_action,
		max_action, 
		hidden_dim=256, 
		exploration_noise=0.1,	#sigma
		discount=0.99,	#gamma 
		tau=0.005,
		actor_lr=3e-4, 
		critic_lr=3e-4, 
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
		if_use_huber_loss=False
	):
		self.device = device
		self.max_action = torch.tensor(max_action).to(device)
		self.min_action = torch.tensor(min_action).to(device)
		
		self.actor = Actor(state_dim, hidden_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

		self.critic = Critic(state_dim, hidden_dim, action_dim, max_action).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

		self.discount = discount
		self.tau = tau
		self.exploration_noise = exploration_noise * (self.max_action - self.min_action) / 2.0

		# Instantiation the loss class (Huber or MSE)
		if if_use_huber_loss:
			self.criterion = torch.nn.SmoothL1Loss(reduction="none")
		else:
			self.criterion = torch.nn.MSELoss(reduction="mean")


	def take_action(self, state, explore=True):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = self.actor(state)
		if explore:
			noise = torch.randn_like(action) * self.exploration_noise
			action = (action + noise).clamp(self.min_action, self.max_action)
		action = action.cpu().data.numpy().flatten()
		return action


	def soft_update(self, net, target_net):
		for param, target_param in zip(net.parameters(), target_net.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def train(self, replay_buffer, batch_size=256):
		# Sample replay buffer 
		tree_idx, batch_memory, ISWeights = replay_buffer.sample_batch(batch_size)

		# Unpack batch_memory into separate lists
		states, actions, rewards, next_states, dones = zip(*batch_memory)

		# Convert lists to NumPy arrays, and then convert to tensors
		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards)
		next_states = np.array(next_states)
		dones = np.array(dones)

		states = torch.tensor(states, dtype=torch.float32, device=self.device)
		actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
		rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # Add an extra dimension
		next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
		dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # Add an extra dimension

		ISWeights = torch.tensor(ISWeights, dtype=torch.float32, device=self.device)

		# Compute the target Q value
		next_Q = self.critic_target(next_states, self.actor_target(next_states))
		target_Q = rewards + ((1 - dones) * self.discount * next_Q)

		# Get current Q estimate
		current_Q = self.critic(states, actions)

		# update priority
		new_priorities = torch.abs(target_Q - current_Q)
		replay_buffer.batch_update(tree_idx=tree_idx, abs_errors=new_priorities.detach().cpu().numpy().squeeze()) 

		# Compute critic loss; from ElegantRL
		td_errors = self.criterion(current_Q, target_Q)
		critic_loss = (td_errors * ISWeights).mean()

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss; elegantRL: use target critic
		actor_loss = -self.critic(states, self.actor(states)).mean()

		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		self.soft_update(self.critic, self.critic_target)  # 软更新价值网络
		self.soft_update(self.actor, self.actor_target)  # 软更新策略网络


	def save(self, filename, save_dir):
		torch.save(self.critic.state_dict(), save_dir + "/critic%d.pth"%(filename))
		torch.save(self.critic_optimizer.state_dict(), save_dir + "/critic_optimizer%d.pth"%(filename))
		
		torch.save(self.actor.state_dict(), save_dir + "/actor%d.pth"%(filename))
		torch.save(self.actor_optimizer.state_dict(), save_dir + "/actor_optimizer%d.pth"%(filename))


	def load(self, filename, save_dir):
		self.critic.load_state_dict(torch.load(save_dir + "/critic%d.pth"%(filename)))
		self.critic_optimizer.load_state_dict(torch.load(save_dir + "/critic_optimizer%d.pth"%(filename)))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(save_dir + "/actor%d.pth"%(filename)))
		self.actor_optimizer.load_state_dict(torch.load(save_dir + "/actor_optimizer%d.pth"%(filename)))
		self.actor_target = copy.deepcopy(self.actor)


	def save_experiment(self, target_folder, i):
		# 使用循环来处理文件名和保存过程以减少代码重复，提高代码的可维护性和可读性。
		# Define the base names and corresponding objects to save
		file_names = [
			(f"critic{i}.pth", self.critic.state_dict()),
			(f"critic_optimizer{i}.pth", self.critic_optimizer.state_dict()),
			(f"actor{i}.pth", self.actor.state_dict()),
			(f"actor_optimizer{i}.pth", self.actor_optimizer.state_dict())
		]
		
		# Save each file
		for file_name, state_dict in file_names:
			target_file = os.path.join(target_folder, file_name)
			torch.save(state_dict, target_file)