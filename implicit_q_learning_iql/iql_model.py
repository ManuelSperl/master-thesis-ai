# iql_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import importlib

import implicit_q_learning_iql.critic
import implicit_q_learning_iql.actor
import implicit_q_learning_iql.value

importlib.reload(implicit_q_learning_iql.critic)
importlib.reload(implicit_q_learning_iql.actor)
importlib.reload(implicit_q_learning_iql.value)

from implicit_q_learning_iql.critic import CriticNet
from implicit_q_learning_iql.actor import ActorNet
from implicit_q_learning_iql.value import ValueNet

class IQLModel(nn.Module):
    def __init__(self, action_dim, device, global_seed, trial_idx):
        super(IQLModel, self).__init__()
        self.device = device
        self.actor = ActorNet(action_dim, global_seed, trial_idx).to(device)
        self.critic1 = CriticNet(action_dim, global_seed, trial_idx).to(device)
        self.critic2 = CriticNet(action_dim, global_seed, trial_idx + 100).to(device)  # Different seed for Critic 2
        self.value = ValueNet(global_seed, trial_idx).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-4)

    def get_action(self, state, eval=False):
        """
        Get an action using the actor network.

        :param state: Input state (tensor)
        :param eval: If True, select the most probable action instead of sampling
        :return: Selected action
        """
        with torch.no_grad():
            return self.actor.get_action(state, eval=eval)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states, actions, rewards, next_states, dones = (
            states.to(self.device), actions.to(self.device),
            rewards.to(self.device), next_states.to(self.device),
            dones.to(self.device)
        )

        # 1️⃣ Compute Next-State Value
        next_value = self.value(next_states)

        # 2️⃣ Compute Critic Loss
        critic1_loss, critic1_pred_q, critic1_target_q = self.critic1.compute_critic_loss(states, actions, rewards, dones, next_value, gamma=0.99)
        critic2_loss, critic2_pred_q, critic2_target_q = self.critic2.compute_critic_loss(states, actions, rewards, dones, next_value, gamma=0.99)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 3️⃣ Compute Value Loss
        value_loss = self.value.compute_value_loss(states, actions, self.critic1, self.critic2, expectile=0.95)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 4️⃣ Compute Actor Loss (Advantage-Weighted Regression)
        with torch.no_grad():
            target_q1 = self.critic1(states, actions)
            target_q2 = self.critic2(states, actions)
            min_Q = torch.min(target_q1, target_q2)
            current_value = self.value(states)

            advantage = (min_Q - current_value)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)  # Normalize
            advantage = torch.clamp(advantage, min=-10, max=10)  # Prevent extreme values

            temperature = 0.2  # Try increasing this value
            exp_advantage = torch.exp(advantage / temperature)
            exp_advantage = torch.clamp(exp_advantage, max=20.0)  # Reduce scale

        # Convert logits into probabilities - Compute actor loss
        logits = self.actor(states)
        log_probs = F.log_softmax(logits, dim=-1) + 1e-6  # log(0) issue +1e-6
        log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # Extract selected action log-prob

        # Use correct weighting in actor loss
        actor_loss = -torch.mean(exp_advantage * log_probs)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return (
            actor_loss.item(),
            critic1_loss.item(), critic1_pred_q, critic1_target_q,
            critic2_loss.item(), critic2_pred_q, critic2_target_q,
            value_loss.item()
        )

