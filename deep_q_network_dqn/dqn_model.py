# dqn_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class DQNModel(nn.Module):
    def __init__(self, action_dim, global_seed, trial_idx):
        super(DQNModel, self).__init__()
        self.global_seed = global_seed

        # Training epsilon (decays over time)
        self.epsilon = 0.1
        self.decay_factor = 0.99
        self.min_epsilon = 0.01

        # Evaluation epsilon (always 0 for greedy policy)
        self.eval_epsilon = 0.0

        # CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_input_dim = self._get_conv_output_dim()

        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.q_values = nn.Linear(512, action_dim)

        self.init_weights(trial_idx)

    def _get_conv_output_dim(self):
        dummy_input = torch.zeros(1, 3, 210, 160)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)

    def init_weights(self, trial_idx):
        seed = self.global_seed + (trial_idx * 1234)
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.q_values(x)

    def get_action(self, state, eval=False):
        """Epsilon-greedy action selection with decay."""
        if not eval:
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.q_values.out_features)  # Random action
            
            # Decay epsilon over time
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_factor)

        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values, dim=1).item()