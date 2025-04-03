# seaquest_dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset

class SeaquestDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list): List of data samples, where each sample is a tuple
                         (state, action, reward, next_state, done).
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the data sample
        obs, action, reward, next_obs, done = self.data[idx]

        # Process observations (normalize and permute)
        obs = torch.tensor(obs, dtype=torch.float32) / 255.0  # Normalize pixel values to [0, 1]
        obs = obs.permute(2, 0, 1)  # (channels, height, width)

        next_obs = torch.tensor(next_obs, dtype=torch.float32) / 255.0
        next_obs = next_obs.permute(2, 0, 1)

        # Convert action to torch.long (integers)
        action = torch.tensor(action, dtype=torch.long)

        # Convert reward to float32
        reward = torch.tensor(reward, dtype=torch.float32)

        # Convert done to float32 or torch.bool
        done = torch.tensor(done, dtype=torch.float32)

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, done