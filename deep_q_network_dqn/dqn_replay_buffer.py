# dqn_replay_buffer.py

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device (torch.device): device to store and sample experiences
        """
        self.device = device
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Store new experience **ONLY IN CPU MEMORY**."""
        state = torch.tensor(state, dtype=torch.float32, device="cpu")  # Store on CPU
        action = torch.tensor(action, dtype=torch.long, device="cpu")
        reward = torch.tensor(reward, dtype=torch.float32, device="cpu")
        next_state = torch.tensor(next_state, dtype=torch.float32, device="cpu")
        done = torch.tensor(done, dtype=torch.float32, device="cpu")

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory. Move samples **to GPU** ONLY during training."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.stack([e.action for e in experiences]).to(self.device).squeeze()
        rewards = torch.stack([e.reward for e in experiences]).to(self.device).squeeze()
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
        dones = torch.stack([e.done for e in experiences]).to(self.device).squeeze()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)