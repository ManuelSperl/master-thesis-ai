import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# ensure the module is re-imported after changes
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
    """
    A class representing the IQLModel which coordinates the actor, critic, and value networks for Implicit Q-Learning.
    """
    def __init__(self, state_size, action_size, learning_rate, hidden_size, tau, temperature, expectile, device, global_seed, trial_idx):
        """
        Initializes the IQLModel with all necessary networks and optimizers.

        :param state_size: dimensionality of the state space
        :param action_size: dimensionality of the action space
        :param learning_rate: learning rate for all optimizers
        :param hidden_size: number of neurons in each hidden layer
        :param tau: soft update parameter
        :param temperature: scaling factor for adjusting the relative importance of entropy
        :param expectile: expectile value for robust value estimation
        :param device: computation device (CPU or GPU)
        :param trial_idx: index of the trial for reproducibility
        """
        super(IQLModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.tau = tau
        self.device = device
        self.global_seed = global_seed  # save the global seed as an instance variable

        self.clip_grad_param = 1
        self.temperature = torch.FloatTensor([temperature]).to(device)
        self.expectile = torch.FloatTensor([expectile]).to(device)
        self.gamma = torch.FloatTensor([0.99]).to(device)

        # ------------------ Setup networks ------------------
        # Actor Network and Optimizer
        self.actor_network = ActorNet(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            global_seed=global_seed,
            trial_idx=trial_idx
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate)

        # Critic Networks and Optimizers
        self.critic1_network = CriticNet(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            global_seed=global_seed,
            trial_idx=trial_idx
        ).to(device)
        self.critic1_target_network = CriticNet(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            global_seed=global_seed,
            trial_idx=trial_idx
        ).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1_network.parameters(), lr=learning_rate)

        self.critic2_network = CriticNet(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            global_seed=global_seed,
            trial_idx=trial_idx + 100 # to get different one then the Critic1
        ).to(device)
        self.critic2_target_network = CriticNet(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            global_seed=global_seed,
            trial_idx=trial_idx
        ).to(device)
        self.critic2_optimizer = optim.Adam(self.critic2_network.parameters(), lr=learning_rate)

        # Value Network and Optimizer
        self.value_network = ValueNet(
            state_size=state_size,
            hidden_size=hidden_size,
            global_seed=global_seed,
            trial_idx=trial_idx
        ).to(device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)

    def get_action(self, state, eval=False):
        """
        Retrieves an action using the policy defined by the actor network.

        :param state: the current state
        :param eval: boolean flag to determine whether the action is deterministic

        :return: the selected action as a numpy array
        """
        # convert state to tensor and move to device
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        else:
            state = state.to(self.device)

        # get action from actor network
        with torch.no_grad():
            if eval:
                action = self.actor_network.get_action(state, eval=True)
            else:
                action = self.actor_network.get_action(state)

        # move action to CPU and convert to numpy
        return action.cpu().numpy()


    def update_target_networks(self, source_model , target_model):
        """
        Updates the target networks using a soft update strategy.

        :param source_model: the source model from which weights are copied
        :param target_model: the target model to which weights are copied
        """
        # update target network parameters using soft update
        for target_param, local_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


    def learn(self, experiences):
        """
        Conducts a learning update using the given batch of experience tuples.

        :param experiences: tuple containing everything

        :return: tuple of loss values for the actor, critic1, critic2, and value network
        """
        states, actions, rewards, next_states, dones = experiences # unpack experience tuple

        # ---------------------- Actor Loss ----------------------

        self.actor_optimizer.zero_grad()
        actor_loss = self.actor_network.compute_actor_loss(
            states=states,
            actions=actions,
            value_network=self.value_network,
            critic1_target_network=self.critic1_target_network,
            critic2_target_network=self.critic2_target_network,
            temperature=self.temperature
        )
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- Critic Losses ----------------------

        # compute next value for target Q calculation
        next_value = self.value_network(next_states)

        # compute Critic 1 Loss
        self.critic1_optimizer.zero_grad()
        critic1_loss, critic1_pred_q, critic1_target_q = self.critic1_network.compute_critic_loss(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_value=next_value,
            gamma=self.gamma
        )
        critic1_loss.backward()
        clip_grad_norm_(self.critic1_network.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        # compute Critic 2 Loss
        self.critic2_optimizer.zero_grad()
        critic2_loss, critic2_pred_q, critic2_target_q = self.critic2_network.compute_critic_loss(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_value=next_value,
            gamma=self.gamma
        )
        critic2_loss.backward()
        clip_grad_norm_(self.critic2_network.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # update target networks
        self.update_target_networks(self.critic1_network, self.critic1_target_network)
        self.update_target_networks(self.critic2_network, self.critic2_target_network)

        # ---------------------- Value Loss ----------------------

        self.value_optimizer.zero_grad()
        value_loss = self.value_network.compute_value_loss(
            states=states,
            actions=actions,
            critic1_target_network=self.critic1_target_network,
            critic2_target_network=self.critic2_target_network,
            expectile=self.expectile
        )
        value_loss.backward()
        self.value_optimizer.step()

        # return losses and Q-values for logging or further processing
        return actor_loss.item(), critic1_loss.item(), critic1_pred_q, critic1_target_q, critic2_loss.item(), critic2_pred_q, critic2_target_q, value_loss.item()