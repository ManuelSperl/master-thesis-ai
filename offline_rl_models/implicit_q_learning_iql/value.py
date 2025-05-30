# value.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    """
    A class representing the value network for Implicit Q-Learning.
    This network estimates the state value function independent of the actions.
    """
    def __init__(self, global_seed, trial_idx):
        """
        Initializes the ValueNet.

        :param global_seed: global seed for reproducibility
        :param trial_idx: index of the trial for reproducibility
        """
        super().__init__()
        self.global_seed = global_seed  # Save the global seed as an instance variable

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the correct size for the fully connected layer dynamically
        self.fc_input_dim = self._get_conv_output_dim()

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 1)

        # Custom weight initialization with trial index
        self.init_weights(trial_idx)

    def _get_conv_output_dim(self):
        """
        Computes the size of the feature map after the convolutional layers.
        """
        dummy_input = torch.zeros(1, 3, 210, 160)  # Seaquest raw input size
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)  # Flatten and get total size

    def init_weights(self, trial_idx):
        """
        Initialize weights with a fixed seed to ensure reproducibility.

        :param trial_idx: index of the trial for reproducibility
        """
        seed = self.global_seed + (trial_idx * 1234)
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight)  # Orthogonal initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Forward pass of the model.

        :param state: input state

        :return: estimated value of the state
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def compute_value_loss(self, states, actions, critic1, critic2, expectile):
        with torch.no_grad():
            target_q1 = critic1(states, actions)
            target_q2 = critic2(states, actions)
            min_Q = torch.min(target_q1, target_q2)  # Use min Q value for conservative estimation

        current_value = self.forward(states)  # Current state values

        # Use normalized expectile loss
        diff = min_Q - current_value
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        loss = (weight * diff.pow(2)).mean()  # Expectile loss

        return loss 

