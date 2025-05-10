# iql_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import importlib

import agent_methods.implicit_q_learning_iql.critic
import agent_methods.implicit_q_learning_iql.actor
import agent_methods.implicit_q_learning_iql.value

importlib.reload(agent_methods.implicit_q_learning_iql.critic)
importlib.reload(agent_methods.implicit_q_learning_iql.actor)
importlib.reload(agent_methods.implicit_q_learning_iql.value)

from agent_methods.implicit_q_learning_iql.critic import CriticNet
from agent_methods.implicit_q_learning_iql.actor import ActorNet
from agent_methods.implicit_q_learning_iql.value import ValueNet

class IQLModel(nn.Module):
    def __init__(self, action_dim, device, global_seed, seed_idx, tau=0.005):
        super(IQLModel, self).__init__()
        self.device = device
        self.tau = tau  # Soft update coefficient

        self.actor = ActorNet(action_dim, global_seed, seed_idx).to(device)
        self.critic1 = CriticNet(action_dim, global_seed, seed_idx).to(device)
        self.critic2 = CriticNet(action_dim, global_seed, seed_idx + 100).to(device)  # Different seed for Critic 2
        self.value = ValueNet(global_seed, seed_idx).to(device)
        self.value_target = ValueNet(global_seed, seed_idx).to(device)
        self.value_target.load_state_dict(self.value.state_dict())  # Initialize target with main value network
        self.value_target.eval()  # Set target network to eval mode

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-5)  # Lowered from 1e-4
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=5e-5)  # Lowered from 1e-4
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=5e-5)  # Lowered from 1e-4
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=5e-5)  # Lowered from 1e-4

    def get_action(self, state, eval=False):
        """
        Get an action using the actor network.

        :param state: Input state (tensor)
        :param eval: If True, select the most probable action instead of sampling
        :return: Selected action
        """
        with torch.no_grad():
            return self.actor.get_action(state, eval=eval)
        
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def learn(self, experiences):
        torch.autograd.set_detect_anomaly(False)  # Debug aid — disable in production

        states, actions, rewards, next_states, dones = experiences
        states, actions, rewards, next_states, dones = (
            states.to(self.device), actions.to(self.device),
            rewards.to(self.device), next_states.to(self.device),
            dones.to(self.device)
        )

        # 1. Compute Next-State Value using target value network
        with torch.no_grad():
            next_value = self.value_target(next_states)

        # 2. Compute Critic Loss
        critic1_loss, critic1_pred_q, critic1_target_q = self.critic1.compute_critic_loss(states, actions, rewards, dones, next_value, gamma=0.99)
        critic2_loss, critic2_pred_q, critic2_target_q = self.critic2.compute_critic_loss(states, actions, rewards, dones, next_value, gamma=0.99)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), 10.0)  # Stability fix
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), 10.0)  # Stability fix
        self.critic2_optimizer.step()

        # 3. Compute Value Loss
        value_loss = self.value.compute_value_loss(states, actions, self.critic1, self.critic2, expectile=0.7)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        clip_grad_norm_(self.value.parameters(), 10.0)  # Stability fix
        self.value_optimizer.step()

        # 4. Compute Actor Loss (Advantage-Weighted Regression)
        with torch.no_grad():
            target_q1 = self.critic1(states, actions)
            target_q2 = self.critic2(states, actions)
            min_q = torch.min(target_q1, target_q2)
            current_value = self.value(states)
            advantage = min_q - current_value

            # Stability Fix: clamp advantage range & temperature raised to 0.1 (was 0.03, in paper)
            advantage = torch.clamp(advantage, min=-5.0, max=5.0)
            positive_advantage_mask = (advantage > 0).float()
            exp_advantage = torch.exp(advantage / 0.1) * positive_advantage_mask

        logits = self.actor(states)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        actor_loss = -torch.mean(exp_advantage * selected_log_probs)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 10.0)  # Stability fix
        self.actor_optimizer.step()

        # 5. Soft update of target value network
        self.soft_update(self.value, self.value_target)

        return (
            actor_loss.item(),
            critic1_loss.item(), critic1_pred_q, critic1_target_q,
            critic2_loss.item(), critic2_pred_q, critic2_target_q,
            value_loss.item()
        )

