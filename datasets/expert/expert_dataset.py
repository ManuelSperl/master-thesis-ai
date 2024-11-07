# expert_dataset.py

import numpy as np
import random
from tqdm import tqdm

# ensure the module is re-imported after changes
import importlib

import datasets.dataset_utils
importlib.reload(datasets.dataset_utils)

from datasets.dataset_utils import save_dataset

def generate_expert_dataset_seaquest(env, model, num_episodes=100, perturbation=False, perturbation_level=0.05, save_path='', file_name=''):
    """
    Generates a dataset using an expert policy derived from a trained model in the Seaquest environment,
    with optional perturbations to simulate suboptimal behavior. Ensures approximately `perturbation_level` of actions are perturbed.

    :param env: Gym or VecEnv environment object, pre-initialized (Seaquest environment)
    :param model: trained model (e.g., a PPO agent) that will decide the actions to take at each step
    :param num_episodes: number of episodes to simulate for generating the dataset (Default is 100)
    :param perturbation: bool, whether to add random perturbation to actions
    :param perturbation_level: float, proportion of actions to perturb (default is 0.05, or 5%)
    :param save_path: str, path to save the dataset
    :param file_name: str, name of the file to save the dataset

    :return: list of tuples, each containing a transition (state, action, reward, next_state, done_flag)
             reflecting the decisions made by the (potentially perturbed) policy
    """
    tqdm.write('Generating Expert Dataset for Seaquest...')

    expert_dataset = []  # list to store the dataset

    current_step = 0  # track the current step for perturbation logic
    perturbed_count = 0  # track the number of perturbed steps

    for _ in tqdm(range(num_episodes), desc='Generating dataset'):
        obs, _ = env.reset()  # Gym returns (observation, info)

        done = False
        while not done:
            # predict action using the model
            action, _ = model.predict(obs, deterministic=True)

            # Ensure action is scalar integer
            action = int(action)

            # Check if this step should be perturbed based on perturbation_level probability
            if perturbation and random.random() < perturbation_level:
                # Choose a random action instead of the model's action
                action = env.action_space.sample()  # No need to wrap in an array
                perturbed_count += 1

            step_return = env.step(action)
            new_obs, reward, done = step_return[:3]

            # Append to dataset
            expert_dataset.append((obs, action, reward, new_obs, done))

            # update current state
            obs = new_obs

            # Update current step count
            current_step += 1

    print('Length of expert dataset:', len(expert_dataset))
    print(f'Number of perturbed steps: {perturbed_count}')

    env.close()  # close the environment

    # Save the dataset
    save_dataset(expert_dataset, save_path, file_name)

    return expert_dataset
