import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Reference: https://github.com/boyu-ai/Hands-on-RL/tree/main
'''

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

	def reinitialize(self):
		self.l1.reset_parameters()
		self.l2.reset_parameters()
		self.l3.reset_parameters()


class Critic(nn.Module):
	def __init__(self, critic_input_dim, hidden_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(critic_input_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)
		
	def forward(self, critic_input):
		q = F.relu(self.l1(critic_input))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q

	def reinitialize(self):
		self.l1.reset_parameters()
		self.l2.reset_parameters()
		self.l3.reset_parameters()


class DDPG:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        min_action,
		max_action, 
        critic_input_dim,
        hidden_dim=256,
		exploration_noise=0.1,	#sigma
        actor_lr=3e-4, 
        critic_lr=3e-4, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.device = device
        self.max_action = torch.tensor(max_action).to(device)
        self.min_action = torch.tensor(min_action).to(device)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = Actor(state_dim, hidden_dim, action_dim, max_action).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(critic_input_dim, hidden_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.exploration_noise = exploration_noise * (self.max_action - self.min_action) / 2.0

    def reinitialize(self):
        self.actor.reinitialize()
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic.reinitialize()
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def take_action(self, state, explore=True):
        state = state.reshape(1, -1).clone().detach().to(self.device).float()
        action = self.actor(state)
        if explore:
            noise = torch.randn_like(action) * self.exploration_noise
            action = (action + noise).clamp(self.min_action, self.max_action)
        action = action.cpu().data.numpy().flatten()
        return action

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, i_agent, filename, save_dir):
        torch.save(self.critic.state_dict(), save_dir + "/critic%d_%d.pth"%(filename, i_agent))
        torch.save(self.critic_optimizer.state_dict(), save_dir + "/critic_optimizer%d_%d.pth"%(filename, i_agent))
        
        torch.save(self.actor.state_dict(), save_dir + "/actor%d_%d.pth"%(filename, i_agent))
        torch.save(self.actor_optimizer.state_dict(), save_dir + "/actor_optimizer%d_%d.pth"%(filename, i_agent))

    def load(self, i_agent, filename, save_dir):
        self.critic.load_state_dict(torch.load(save_dir + "/critic%d_%d.pth"%(filename, i_agent)))
        #self.critic_optimizer.load_state_dict(torch.load(save_dir + "/critic_optimizer%d_%d.pth"%(filename, i_agent)))
        self.target_critic = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(save_dir + "/actor%d_%d.pth"%(filename, i_agent)))
        #self.actor_optimizer.load_state_dict(torch.load(save_dir + "/actor_optimizer%d_%d.pth"%(filename, i_agent)))
        self.target_actor = copy.deepcopy(self.actor)


class MADDPG:
    def __init__(
        self, 
        num_agents, 
        state_dims, 
        action_dims, 
        min_action,
		max_action, 
        critic_input_dim, 
        hidden_dim,
		exploration_noise=0.3,
        gamma=0.98, 
        tau=0.005,
        actor_lr=2e-4, 
        critic_lr=4e-4, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.device = device
        self.num_agents = num_agents
        self.agents = []

        for _ in range(self.num_agents):
            self.agents.append(
                DDPG(
                    state_dims, 
                    action_dims, 
                    min_action,
		            max_action, 
                    critic_input_dim,
                    hidden_dim, 
                    exploration_noise,
                    actor_lr, 
                    critic_lr, 
                    device
                )
            )

        self.gamma = gamma
        self.tau = tau
        # Instantiation the MSE loss class
        self.criterion = nn.MSELoss(reduction="mean")

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore=True):
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def train(self, replay_buffer, batch_size, i_agent):
		# Sample replay buffer 
        states, actions, rewards, next_states, dones = replay_buffer.sample_batch(batch_size)
        rewards = rewards.unsqueeze(2)

        cur_agent = self.agents[i_agent]
        
        with torch.no_grad():
            target_actions = torch.zeros_like(actions)
            for i, policy in enumerate(self.target_policies):
                target_actions[:, i] = policy(next_states[:, i])

            target_critic_input = torch.cat((next_states[:, :, :3].reshape(batch_size, -1), 
                                             next_states[:, :, 6:9].reshape(batch_size, -1), 
                                             target_actions[:, :, :].reshape(batch_size, -1)), dim=1)
            next_Q = cur_agent.target_critic(target_critic_input)
            target_critic_value = rewards[:, i_agent] + (1 - dones) * self.gamma * next_Q
            
            critic_input = torch.cat((states[:, :, :3].reshape(batch_size, -1), 
                                      states[:, :, 6:9].reshape(batch_size, -1), 
                                      actions[:, :, :].reshape(batch_size, -1)), dim=1)
        
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.criterion(critic_value, target_critic_value)

        cur_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        cur_actor_outs = cur_agent.actor(states[:, i_agent])
        actions[:, i_agent] = cur_actor_outs
        cur_critic_input = torch.cat((states[:, :, :3].reshape(batch_size, -1), 
                                      states[:, :, 6:9].reshape(batch_size, -1), 
                                      actions[:, :, :].reshape(batch_size, -1)), dim=1)
        actor_loss = -cur_agent.critic(cur_critic_input).mean()

        cur_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

        #self.update_all_targets()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

    def reinitialize(self):
        for agt in self.agents:
            agt.reinitialize()

    def save(self, filename, save_dir):
        for i, agt in enumerate(self.agents):
            agt.save(i, filename, save_dir)

    def load(self, filename, save_dir):
        for i, agt in enumerate(self.agents):
            agt.load(i, filename, save_dir)