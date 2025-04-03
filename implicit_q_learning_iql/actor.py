# actor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorNet(nn.Module):
    def __init__(self, action_dim, global_seed, trial_idx):
        super(ActorNet, self).__init__()
        self.global_seed = global_seed
        self.trial_idx = trial_idx

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Dynamically calculate input size for FC layer
        self.fc_input_dim = self._get_conv_output_dim()

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.action_logits = nn.Linear(512, action_dim)

        # Initialize weights
        self.init_weights()

    def _get_conv_output_dim(self):
        """
        Computes the size of the feature map after the convolutional layers dynamically.
        This ensures compatibility across different input sizes.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 210, 160)  # Example input size (Seaquest)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)  # Flatten and get total size

    def init_weights(self):
        seed = self.global_seed + (self.trial_idx * 1234)  # Unique seed per trial
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
        x = x.view(x.size(0), -1)  # Flatten the tensor dynamically
        x = F.relu(self.fc1(x))
        return self.action_logits(x)

    def get_action(self, state, eval=False, exploration_eps=0.05):
        logits = self.forward(state)
        action_probabilities = F.softmax(logits, dim=-1)
        action_distribution = Categorical(action_probabilities)

        if eval:
            # 🚀 90% argmax, 10% random action for diversity
            if torch.rand(1).item() < exploration_eps:
                return torch.randint(0, action_probabilities.shape[-1], (1,)).to(state.device)
            return torch.argmax(action_probabilities, dim=-1)
        else:
            return action_distribution.sample()

