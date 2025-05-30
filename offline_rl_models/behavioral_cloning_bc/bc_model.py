# bc_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCModel(nn.Module):
    """
    A class representing the Behavior Cloning model.
    """
    def __init__(self, action_dim, global_seed, seed_idx):
        """
        Initializes the BCModel.

        :param action_dim: dimensionality of the action space
        :param global_seed: global seed for reproducibility
        :param seed_idx: index of the seed for reproducibility
        """
        super(BCModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the size of the feature map after the conv layers
        self.fc_input_dim = self._get_conv_output_dim()

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, action_dim)

        self.global_seed = global_seed  # Save the global seed as an instance variable

        # Custom weight initialization with seed index
        self.init_weights(seed_idx)

    def _get_conv_output_dim(self):
        # Create a dummy input to pass through conv layers
        dummy_input = torch.zeros(1, 3, 210, 160)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        output_dim = int(np.prod(x.size()[1:]))  # Exclude batch size
        return output_dim

    def forward(self, state):
        """
        Forward pass of the model.

        :param state: input state

        :return: predicted action logits
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        action_logits = self.fc2(x)
        return action_logits

    def init_weights(self, seed_idx):
        """
        Initialize weights with a fixed seed to ensure reproducibility.

        :param seed_idx: index of the seed for reproducibility
        """
        seed = self.global_seed + (seed_idx * 1234)  # Set seed differently for each seed

        # Initialize weights with seed
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
