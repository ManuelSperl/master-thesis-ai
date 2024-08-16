# ----- PyTorch imports -----
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCModel(nn.Module):
    """
    A class representing the Behavior Cloning model.
    """
    def __init__(self, state_dim, action_dim, global_seed, trial_idx):
        """
        Initializes the BCModel.

        :param state_dim: dimensionality of the state space
        :param action_dim: dimensionality of the action space
        :param trial_idx: index of the trial for reproducibility
        """
        super(BCModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)

        self.global_seed = global_seed  # save the global seed as an instance variable

        # custom weight initialization with trial index
        self.init_weights(trial_idx)

    def forward(self, state):
        """
        Forward pass of the model.

        :param state: input state

        :return: predicted action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = self.fc4(x)

        return action


    def init_weights(self, trial_idx):
        """
        Initialize weights with a fixed seed to ensure reproducibility.

        :param trial_idx: index of the trial for reproducibility
        """
        seed = self.global_seed + (trial_idx * 1234) # set seed different for every trial, but consistent for reproducability

        # initialize weights with seed
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.manual_seed(seed)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)