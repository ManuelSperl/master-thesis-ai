# critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNet(nn.Module):
    """
    A class representing the critic network in Implicit Q-Learning.
    """
    def __init__(self, action_dim, global_seed, trial_idx):
        super().__init__()
        self.global_seed = global_seed
        self.action_dim = action_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute dynamic feature size for fc1
        self.fc_input_dim = self._get_conv_output_dim()

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)

        # Custom weight initialization
        self.init_weights(trial_idx)

    def _get_conv_output_dim(self):
        """
        Computes the size of the feature map after the convolutional layers.
        """
        dummy_input = torch.zeros(1, 3, 210, 160)  # Seaquest input size
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)  # Flatten and get size

    def init_weights(self, trial_idx):
        """
        Initialize weights with a fixed seed for reproducibility.
        """
        seed = self.global_seed + (trial_idx * 1234)
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight)  # Orthogonal initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        """
        Forward pass for computing Q-values.

        :param state: input state
        :param action: action taken in the state

        :return: Q-value predicting the expected reward
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Convert action to one-hot and concatenate with features
        action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        x = torch.cat((x, action_one_hot), dim=1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def compute_critic_loss(self, states, actions, rewards, dones, next_value, gamma):
        current_q_value = self.forward(states, actions)

        # Clamp next_value to avoid extreme updates
        next_value = torch.clamp(next_value, min=-100, max=100)

        q_target = rewards + gamma * (1 - dones) * next_value.detach()

        # Clamp Q targets to stabilize learning
        q_target = torch.clamp(q_target, min=-300, max=300)

        loss = F.smooth_l1_loss(current_q_value, q_target)

        return loss, current_q_value, q_target
