import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from acoustorl.common.general_utils import ReplayBuffer

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# Multiple update times: add "update times" or "Gradient steps" (no guradually increasing).


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3():
	def __init__(
		self,
		state_dim,
		action_dim,
		min_action,
		max_action,
		exploration_noise=0.1,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		replay_size=1e6,
		repeat_times=1.0,
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.replay_size = replay_size
		self.device = device
		self.max_action = torch.tensor(max_action).to(device)
		self.min_action = torch.tensor(min_action).to(device)

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discount = discount
		self.tau = tau
		self.exploration_noise = exploration_noise
		self.policy_noise = policy_noise * (self.max_action - self.min_action) / 2.0
		self.noise_clip = noise_clip * (self.max_action - self.min_action) / 2.0
		self.policy_freq = policy_freq

		self.total_it = 0
		self.repeat_times = repeat_times

		self.replay_buffer = ReplayBuffer(state_dim = self.state_dim,
                                          action_dim = self.action_dim,
                                          max_size = self.replay_size,
                                          device = self.device)
		
		# Instantiation the MSE loss class
		self.criterion = torch.nn.MSELoss(reduction="mean")


	def take_action(self, state, explore=True):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = self.actor(state)
		if explore:
			noise = torch.randn_like(action) * self.exploration_noise
			action = (action + noise).clamp(self.min_action, self.max_action)
		action = action.cpu().data.numpy().flatten()
		return action
	

	def train(self, batch_size=256):
		update_times = self.repeat_times
		for t in range(update_times):
			self.total_it += 1

			# Sample replay buffer (get tensors) (reward: torch.Size([256, 1]))
			state, action, reward, next_state, done = self.replay_buffer.sample_batch(batch_size)

			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)
				
				next_action = (
					self.actor_target(next_state) + noise
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(next_state, next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + (1-done) * self.discount * target_Q

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss (TD errors)
			critic_loss = self.criterion(current_Q1, target_Q) + self.criterion(current_Q2, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if self.total_it % self.policy_freq == 0:

				# Compute actor losse
				actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
				
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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