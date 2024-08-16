import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNet(nn.Module):
    """
    A class representing the critic network in an actor-critic framework, for Implicit Q-Learning.
    This network evaluates the quality of actions taken by the actor.
    """
    def __init__(self, state_size, action_size, hidden_size, global_seed, trial_idx):
        """
        Initializes the CriticNet.

        :param state_size: dimensionality of the state space
        :param action_size: dimensionality of the action space
        :param hidden_size: number of neurons in each hidden layer
        :param trial_idx: index of the trial for reproducibility
        """
        super(CriticNet, self).__init__()
        self.global_seed = global_seed  # save the global seed as an instance variable

        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # custom weight initialization with trial index (for reproducibility)
        self.init_weights(trial_idx)

    def forward(self, state, action):
        """
        Forward pass of the model that computes Q-values.

        :param state: input state
        :param action: action taken in the state

        :return: Q-value predicting the expected reward
        """
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    def init_weights(self, trial_idx):
        """
        Initialize weights with a fixed seed to ensure reproducibility.

        :param trial_idx: index of the trial for reproducibility
        """
        # set seed different for every trial, but consistent for reproducability
        seed = self.global_seed + (trial_idx * 1234)
        #print("Critic seed: ", seed) # to check seed

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.manual_seed(seed)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_critic_loss(self, states, actions, rewards, dones, next_value, gamma):
        """
        Computes the loss for the critic network using Temporal Difference (TD) error.

        :param states: states visited by the agent
        :param actions: actions taken by the agent
        :param rewards: rewards received after taking actions
        :param dones: boolean flags indicating if a state is terminal
        :param next_value: value of the next state as predicted by the target network
        :param gamma: discount factor for future rewards

        :return: loss, Q-values predicted by the critic, and the target Q-values
        """
        # calculate the target Q-value
        with torch.no_grad():
            q_target = rewards + (gamma * (1 - dones) * next_value)

        # calculate the current Q-value
        current_q_value = self.forward(states, actions)

        # calculate the loss
        loss = F.mse_loss(current_q_value, q_target)

        # return both the loss and the Q-values
        return loss, current_q_value, q_target