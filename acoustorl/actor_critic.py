import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference: https://github.com/kaixindelele/DRLib/blob/main/memory/sp_per_memory_torch.py
# define the actor network and the critic network


class Actor(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim, max_action):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)
		
		self.max_action = max_action


	def forward(self, state, action):
        # action should be a tensor, not a list of tensors
		action /= self.max_action
		sa = torch.cat([state, action], dim=1)

		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q