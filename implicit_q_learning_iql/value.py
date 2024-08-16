import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    """
    A class representing the value network for Implicit Q-Learning.
    This network estimates the state value function independent of the actions.
    """
    def __init__(self, state_size, hidden_size, global_seed, trial_idx):
        """
        Initializes the ValueNet.

        :param state_size: dimensionality of the state space
        :param hidden_size: number of neurons in each hidden layer
        :param trial_idx: index of the trial for reproducibility
        """
        super(ValueNet, self).__init__()
        self.global_seed = global_seed  # save the global seed as an instance variable

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # custom weight initialization with trial index (for reproducibility)
        self.init_weights(trial_idx)

    def forward(self, state):
        """
        Forward pass of the model.

        :param state: input state

        :return: estimated value of the state
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    def init_weights(self, trial_idx):
        """
        Initialize weights with a fixed seed to ensure reproducibility.

        :param trial_idx: index of the trial for reproducibility
        """
        # set seed different for every trial, but consistent for reproducability
        seed = self.global_seed + (trial_idx * 1234)
        #print("Value seed: ", seed) # to check seed

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.manual_seed(seed)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def expectile_loss(self, diff, expectile=0.8):
        """
        Calculates the expectile loss, which is used for robust value estimation.

        :param diff: difference between target Q values and estimated values
        :param expectile: expectile coefficient

        :return: calculated expectile loss
        """
        # calculate the weight based on the difference
        weight = torch.where(diff > 0, expectile, (1 - expectile))

        # calculate and return the expectile loss
        return weight * (diff**2)

    def compute_value_loss(self, states, actions, critic1_target_network, critic2_target_network, expectile):
        """
        Computes the value loss using the expectile loss function.

        :param states: input states
        :param actions: actions taken at the states
        :param critic1_target_network: first critic target network
        :param critic2_target_network: second critic target network
        :param expectile: expectile coefficient

        :return: computed value loss
        """
        # get target Q values
        with torch.no_grad():
            target_q1  = critic1_target_network(states, actions)
            target_q2 = critic2_target_network(states, actions)
            min_Q = torch.min(target_q1, target_q2)

        # get current value
        current_value = self.forward(states)

        # calculate the expectile loss
        value_loss = self.expectile_loss(min_Q - current_value, expectile).mean()

        return value_loss