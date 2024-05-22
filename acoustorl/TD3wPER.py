import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from acoustorl.common.per import ReplayBuffer

# Modified from TD3wPP_3.py
# Using per.py, PrioritizedReplayBuffer
# Optimized function train(): use Huber loss


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
	

	def reinitialize(self):
		self.l1.reset_parameters()
		self.l2.reset_parameters()
		self.l3.reset_parameters()


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


	def reinitialize(self):
		self.l1.reset_parameters()
		self.l2.reset_parameters()
		self.l3.reset_parameters()
		self.l4.reset_parameters()
		self.l5.reset_parameters()
		self.l6.reset_parameters()


class TD3():
	def __init__(
		self,
		state_dim,
		action_dim,
		min_action,
		max_action,
		exploration_noise=0.1,
		policy_noise=0.2,		
		tau=0.005,
		discount=0.99,
		noise_clip=0.5,
		policy_freq=2,
		replay_size=1e6,
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.replay_size = replay_size
		self.device = device
		self.max_action = torch.tensor(max_action).to(device)
		self.min_action = torch.tensor(min_action).to(device)

		self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=2e-4)

		self.critic = Critic(self.state_dim, self.action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4)

		self.discount = discount
		self.tau = tau
		self.exploration_noise = exploration_noise
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

		self.replay_buffer = ReplayBuffer(obs_dim = self.state_dim,
                                          act_dim = self.action_dim,
                                          size = self.replay_size,
                                          device = self.device)
		
		# from ElegantRL
		self.criterion = torch.nn.SmoothL1Loss(reduction="none")


	def reinitialize(self):
		self.actor.reinitialize()
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic.reinitialize()
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)


	def reinitialize_replay_buffer(self):
		self.replay_buffer = ReplayBuffer(obs_dim = self.state_dim,
									      act_dim = self.action_dim,
									      size = self.replay_size,
									      device = self.device)


	def take_action(self, state, explore=True):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = self.actor(state)
		if explore:
			noise = torch.randn_like(action) * self.exploration_noise
			action = (action + noise).clamp(self.min_action, self.max_action)
		action = action.cpu().data.numpy().flatten()
		return action


	def train(self, batch_size):
		self.total_it += 1

		# Sample replay buffer 
		data = self.replay_buffer.sample_batch(batch_size)

		tree_idx, batch_memory, ISWeights = data

		# Unpack batch_memory into separate lists
		states, actions, rewards, next_states, dones = map(list, zip(*batch_memory))

		# Convert lists to NumPy arrays
		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards)
		next_states = np.array(next_states)
		dones = np.array(dones)
		ISWeights = np.array(ISWeights)

		# Convert lists to tensors
		states = torch.tensor(states, dtype=torch.float32, device=self.device)
		actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
		rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # Add an extra dimension
		next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
		dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # Add an extra dimension
		ISWeights = torch.tensor(ISWeights, dtype=torch.float32, device=self.device)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(actions) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_states) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_states, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = rewards + (1-dones)*self.discount*target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(states, actions)

		# from ElegantRL
		td_errors = self.criterion(current_Q1, target_Q) + self.criterion(current_Q2, target_Q)
		# update priority
		self.replay_buffer.batch_update(tree_idx=tree_idx, abs_errors=td_errors.detach().cpu().numpy().squeeze())  
	
		# Compute critic loss; from ElegantRL
		critic_loss = (td_errors * ISWeights).mean()

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss; elegantRL: use target critic
			actor_loss = -self.critic_target.Q1(states, self.actor(states)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def weighted_mse_loss(self, input, target, weight):
		return torch.sum(weight * (input - target) ** 2).mean()


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


	def save_experiment(self, env_name, i):
		torch.save(self.critic.state_dict(), "TD3_%s_data/critic%d.pth"%(env_name, i))
		torch.save(self.critic_optimizer.state_dict(), "TD3_%s_data/critic_optimizer%d.pth"%(env_name, i))
		
		torch.save(self.actor.state_dict(), "TD3_%s_data/actor%d.pth"%(env_name, i))
		torch.save(self.actor_optimizer.state_dict(), "TD3_%s_data/actor_optimizer%d.pth"%(env_name, i))