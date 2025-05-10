# bve_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import importlib

import agent_methods.implicit_q_learning_iql.critic
importlib.reload(agent_methods.implicit_q_learning_iql.critic)
from agent_methods.implicit_q_learning_iql.critic import CriticNet

class BVEModel(nn.Module):
    def __init__(self, action_dim, device, global_seed, trial_idx, tau=0.005):
        super(BVEModel, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.tau = tau

        # Reuse the critic network as the Q-network
        self.q_net = CriticNet(action_dim, global_seed, trial_idx).to(device)
        self.target_q_net = CriticNet(action_dim, global_seed, trial_idx + 100).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=5e-5)
        self.gamma = 0.99

    def soft_update(self):
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def forward(self, state, action):
        return self.q_net(state, action)

    def get_action(self, state):
        with torch.no_grad():
            batch_size = state.size(0)
            action_range = torch.arange(self.action_dim, device=state.device)

            # Repeat states for each action
            state_repeat = state.unsqueeze(1).repeat(1, self.action_dim, 1, 1, 1).view(-1, *state.shape[1:])
            actions = action_range.repeat(batch_size)  # [A, A, ..., A]

            # Q(s,a) for all (s,a)
            q_values = self.q_net(state_repeat, actions)
            q_values = q_values.view(batch_size, self.action_dim)

            return torch.argmax(q_values, dim=1)


    def learn(self, experiences):
        torch.autograd.set_detect_anomaly(False)

        states, actions, rewards, next_states, dones = experiences
        states, actions, rewards, next_states, dones = (
            states.to(self.device), actions.to(self.device),
            rewards.to(self.device), next_states.to(self.device),
            dones.to(self.device)
        )

        # Sample next actions using target Q-network's greedy policy
        # Target Q: r + γ Q(s', argmax_a' Q(s', a'))
        with torch.no_grad():
            batch_size = next_states.size(0)
            q_values = []
            for a in range(self.action_dim):
                a_tensor = torch.full((batch_size,), a, device=self.device, dtype=torch.long)
                q = self.target_q_net(next_states, a_tensor).squeeze()
                q_values.append(q.unsqueeze(1))
            q_values = torch.cat(q_values, dim=1)
            next_actions = torch.argmax(q_values, dim=1)
            target_q = self.target_q_net(next_states, next_actions)
            target_q = torch.clamp(target_q, min=-300, max=300)  # match IQL

        current_q = self.q_net(states, actions)
        target = rewards + self.gamma * (1 - dones) * target_q.detach()

        loss = F.smooth_l1_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.soft_update()

        return loss.item(), current_q, target