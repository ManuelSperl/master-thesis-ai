# mixed_dataset.py

import numpy as np
import random
from tqdm import tqdm

# ensure the module is re-imported after changes
import importlib

import datasets.dataset_utils
importlib.reload(datasets.dataset_utils)

from datasets.dataset_utils import save_dataset

def generate_mixed_dataset_seaquest(env, model, num_episodes=100, perturbation=False, perturbation_level=0.05, mix_ratio=0.5, save_path='', file_name=''):
    """
    Generates a mixed dataset where actions are a mix of expert and random actions, with optional perturbations.
    
    :param env: Gym or VecEnv environment object, pre-initialized (Seaquest environment)
    :param model: trained model (e.g., a PPO agent) that will decide the actions to take at each step
    :param num_episodes: number of episodes to simulate for generating the dataset (Default is 100)
    :param perturbation: bool, whether to add random perturbation to actions
    :param perturbation_level: float, proportion of actions to perturb (default is 0.05, or 5%)
    :param mix_ratio: float, proportion of actions to be taken by the expert model (default is 0.5, or 50%)
    :param save_path: str, path to save the dataset
    :param file_name: str, name of the file to save the dataset

    :return: list of tuples, each containing a transition (state, action, reward, next_state, done_flag)
    """
    tqdm.write('Generating Mixed Dataset for Seaquest...')

    mixed_dataset = []
    perturbed_count = 0

    for _ in tqdm(range(num_episodes), desc='Generating dataset'):
        obs, _ = env.reset()

        done = False
        while not done:
            # Decide whether to use the expert action or a random action
            if random.random() < mix_ratio:
                # Expert action
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                # Random action
                action = env.action_space.sample()

            # Perturb action if enabled
            if perturbation and random.random() < perturbation_level:
                action = env.action_space.sample()
                perturbed_count += 1

            step_return = env.step(action)
            new_obs, reward, done = step_return[:3]

            # Append to dataset
            mixed_dataset.append((obs, action, reward, new_obs, done))

            obs = new_obs  # Update current state

    print('Length of mixed dataset:', len(mixed_dataset))
    print(f'Number of perturbed steps: {perturbed_count}')

    env.close()

    # Save the dataset
    save_dataset(mixed_dataset, save_path, file_name)

    return mixed_dataset
