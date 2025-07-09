# seaquest_dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset

class SeaquestDataset(Dataset):
    """
    Custom dataset for the Seaquest environment.
    """
    def __init__(self, data, transform=None):
        """
        Initializes the SeaquestDataset with data and an optional transform.

        :param data: List of transitions, where each transition is a tuple of the form:
            (obs, action, reward, next_obs, next_action, done, perturbed_flag, original_action)
        :param transform: Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single data sample from the dataset.

        :param idx: Index of the sample to retrieve.
        :return: A tuple containing:
            (obs, action, reward, next_obs, next_action, done, perturbed_flag, original_action)
        """
        # get the data sample
        obs, action, reward, next_obs, next_action, done, perturbed, original_action = self.data[idx]

        # process observations (normalize and permute)
        obs = torch.tensor(obs, dtype=torch.float32) / 255.0  # normalize pixel values to [0, 1]
        obs = obs.permute(2, 0, 1)  # (channels, height, width)

        next_obs = torch.tensor(next_obs, dtype=torch.float32) / 255.0
        next_obs = next_obs.permute(2, 0, 1)

        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        next_action = torch.tensor(next_action, dtype=torch.long)
        original_action = torch.tensor(original_action, dtype=torch.long)
        perturbed = torch.tensor(perturbed, dtype=torch.bool)

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, next_action, done, perturbed, original_action