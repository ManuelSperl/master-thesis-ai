import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorNet(nn.Module):
    """
    A class representing the actor network for Implicit Q-Learning.
    This network outputs actions as per a squashed Gaussian policy.
    """
    def __init__(self, state_size, action_size, hidden_size, global_seed, trial_idx, min_log_std=-10, max_lox_std=2):
        """
        Initializes the ActorNet.

        :param state_size: dimensionality of the state space
        :param action_size: dimensionality of the action space
        :param hidden_size: number of neurons in each hidden layer
        :param trial_idx: index of the trial for reproducibility
        :param min_log_std: minimum value for the logarithm of the standard deviation of the Gaussian distribution
        :param max_lox_std: maximum value for the logarithm of the standard deviation of the Gaussian distribution
        """
        super(ActorNet, self).__init__()
        self.min_log_std = min_log_std
        self.max_lox_std = max_lox_std
        self.global_seed = global_seed  # save the global seed as an instance variable

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)

        # custom weight initialization with trial index (for reproducibility)
        self.init_weights(trial_idx)

    def forward(self, state):
        """
        Forward pass of the model.

        :param state: input state

        :return: tuple of mean (mu) and standard deviation (std) of actions
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # calculate mean and standard deviation of actions
        mu = torch.tanh(self.mu(x))
        log_std = self.log_std(x)

        # clamping ensures that standard deviation stays within reasonable bounds during training
        std = torch.clamp(log_std, self.min_log_std, self.max_lox_std)

        return mu, std

    def init_weights(self, trial_idx):
        """
        Initialize weights with a fixed seed to ensure reproducibility.

        :param trial_idx: index of the trial for reproducibility
        """
        # set seed different for every trial, but consistent for reproducability
        seed = self.global_seed + (trial_idx * 1234)
        #print("Actor seed: ", seed) # to check seed

        # initialize weights with seed
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.manual_seed(seed)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def sample_action(self, state, epsilon=1e-6):
        """
        Samples an action from the Gaussian distribution defined by the network outputs.

        :param state: input state
        :param epsilon: small number to ensure numerical stability

        :return: sampled action and action distribution
        """
        mu, log_std = self.forward(state) # get mean and log standard deviation
        std = log_std.exp()
        action_distribution = Normal(mu, std) # create normal distribution
        action = action_distribution.rsample() # sample action

        return action, action_distribution

    def get_action(self, state, eval=False):
        """
        Computes an action using the deterministic or stochastic policy and returns the
        action based on a squashed Gaussian policy.

        :param state: input state
        :param eval: boolean flag to use deterministic policy if True, otherwise sample from the distribution

        :return: action tensor
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        action_distribution = Normal(mu, std)

        # sample action if not in evaluation mode
        if eval:
            action = mu
        else:
            action = action_distribution.rsample()

        return action.detach().cpu()

    def compute_actor_loss(self, states, actions, value_network, critic1_target_network, critic2_target_network, temperature):
        """
        Computes the actor loss for Implicit Q-Learning.

        :param states: input states
        :param actions: input actions
        :param value_network: value network
        :param critic1_target_network: target network for critic 1
        :param critic2_target_network: target network for critic 2
        :param temperature: temperature parameter

        :return: actor loss
        """
        # get current value and target Q values
        with torch.no_grad():
            current_value  = value_network(states)
            target_q1  = critic1_target_network(states, actions)
            target_q2  = critic2_target_network(states, actions)
            min_Q = torch.min(target_q1, target_q2)

        # calculate the exponential adjustment
        exp_adjustment = torch.exp((min_Q - current_value) * temperature)
        exp_adjustment = torch.min(exp_adjustment, torch.FloatTensor([100.0]).to(states.device))

        # sample action and get log probabilities
        _, action_distribution = self.sample_action(states)
        log_probabilities = action_distribution.log_prob(actions)

        # calculate actor loss
        actor_loss = -(exp_adjustment * log_probabilities).mean()

        return actor_loss