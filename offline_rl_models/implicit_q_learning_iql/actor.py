# actor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorNet(nn.Module):
    """
    Actor network for Implicit Q-Learning (IQL) in Atari environments.
    """
    def __init__(self, action_dim, global_seed, trial_idx):
        """
        Initializes the Actor network with convolutional and fully connected layers.

        :param action_dim: Dimension of the action space
        :param global_seed: Global seed for reproducibility
        :param trial_idx: Index of the trial for initializing networks
        """
        super().__init__()
        self.global_seed = global_seed
        self.trial_idx = trial_idx

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # dynamically calculate input size for FC layer
        self.fc_input_dim = self._get_conv_output_dim()

        # fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.action_logits = nn.Linear(512, action_dim)

        # initialize weights
        self.init_weights()

    def _get_conv_output_dim(self):
        """
        Computes the size of the feature map after the convolutional layers dynamically.
        This ensures compatibility across different input sizes.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 210, 160)  # example input size (Seaquest)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)  # flatten and get total size

    def init_weights(self):
        """
        Initialize weights with a fixed seed for reproducibility.
        """
        seed = self.global_seed + (self.trial_idx * 1234)  # unique seed per trial
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Forward pass through the network to compute action logits.

        :param state: Input state tensor
        :return: Action logits tensor
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten the tensor dynamically
        x = F.relu(self.fc1(x))
        return self.action_logits(x)

    def get_action(self, state, eval=False):
        """
        Selects an action based on the current state. If eval is True, it returns the
        action with the highest probability (deterministic policy). Otherwise, it samples
        from the action distribution (stochastic policy).

        :param state: Current state tensor
        :param eval: Whether to use a deterministic policy (default is False)
        """
        logits = self.forward(state)
        action_probabilities = F.softmax(logits, dim=-1)
        action_distribution = Categorical(action_probabilities)

        if eval:
            return torch.argmax(action_probabilities, dim=-1)  # deterministic
        else:
            return action_distribution.sample()  # stochastic

