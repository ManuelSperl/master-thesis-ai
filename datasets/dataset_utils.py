# dataset_utils.py

import pickle
import numpy as np
import random
from tqdm import tqdm
import torch
import os
import gc
import matplotlib.pyplot as plt
from itertools import islice
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import itertools
from functools import partial

import importlib
import datasets.seaquest_dataset
importlib.reload(datasets.seaquest_dataset)

from datasets.seaquest_dataset import SeaquestDataset


def evaluate_dqn_agent(env, model, seed, num_episodes=5):
    """
    Evaluates a DQN agent on a given environment for a specified number of episodes.
    
    :param env: Gym environment to evaluate the agent on
    :param model: Trained DQN model to evaluate
    :param seed: Seed for reproducibility
    :param num_episodes: Number of episodes to run for evaluation
    :return: Average reward over the episodes and action counts
    """
    episode_rewards = []
    action_counts = {}
    episode_idx = 0

    # use tqdm
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        total_rewards = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)

            # convert action to scalar if it's a NumPy array
            if isinstance(action, np.ndarray):
                action = action.item()
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_rewards += reward
            # log action
            action_counts[action] = action_counts.get(action, 0) + 1
            
        episode_rewards.append(total_rewards)
        episode_idx += 1
        
    average_reward = np.mean(episode_rewards)
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")
    print(f"Action Counts: {action_counts}")
    return average_reward, action_counts

def worker_init_fn(worker_id, seed):
    """
    Initializes the random seed for each DataLoader worker.
    This function needs to be at the top level to be picklable.

    :param worker_id: ID of the worker
    :param seed: seed for random state to ensure reproducibility
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def create_environment(env_id, seed=None):
    """
    Creates a raw Gym environment (no wrappers) for offline RL (BC, IQL, BVE) and dataset generation.
    This preserves raw RGB (210x160) observations and does not apply any preprocessing.

    :param env_id: Gym environment ID (e.g., "SeaquestNoFrameskip-v4")
    :param seed: Seed for reproducibility
    :return: Unwrapped gym.Env instance
    """
    env = gym.make(env_id, render_mode=None)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env

def set_all_seeds(seed):
    """
    Function that sets a seed for reproducibility.

    :param seed: the seed to set
    """
    random.seed(seed)  # python random seed
    np.random.seed(seed)  # numPy random seed
    torch.manual_seed(seed)  # pyTorch CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # pyTorch GPU seed
        torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  # avoids non-deterministic optimizations

    set_random_seed(seed)  # for stable_baselines3

def save_dataset(dataset, path, filename):
    """
    Saves a dataset to a specified path and file.

    :param dataset: dataset to save
    :param path: directory where the file will be saved
    :param filename: name of the file to save the dataset to
    """
    # ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    # construct the full file path
    full_path = os.path.join(path, filename)
    
    # save the dataset to the specified file
    with open(full_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {full_path}")

def generate_dataset(env, agent, seed, target_size=150_000, perturbation=False, perturbation_level=0.05, save_path='', file_name=''):
    """
    Generates a dataset with a fixed number of transitions (target_size) using an expert policy.
    Always includes full transition info for statistics and BVE:
        obs, original_action, perturbed_action, reward, next_obs, done, perturbed_flag, next_action

    :param env: Gym or VecEnv environment object (Seaquest)
    :param agent: Trained expert agent (DQN)
    :param seed: Seed for reproducibility
    :param target_size: Total number of transitions to collect
    :param perturbation: Whether to apply random action perturbation
    :param perturbation_level: Proportion of transitions to perturb (e.g., 0.05 for 5%)
    :param save_path: Directory to save the dataset
    :param file_name: File name for the saved dataset (.pkl)
    :return: The full list of transitions
    """
    tqdm.write(f"Generating dataset with at least {target_size} transitions (full episodes only)...")

    dataset = []
    perturbed_count = 0
    transition_counter = 0
    episode_idx = 0

    pbar = tqdm(
        total=target_size,
        desc="Collecting transitions",
        dynamic_ncols=False,
        initial=0,
        bar_format="Collecting transitions: {n_fmt}/150000 [{elapsed}<{remaining}, {rate_fmt}]"
    )

    while transition_counter < target_size:
        obs, _ = env.reset()
        done = False
        episode_transitions = []

        while not done:
            # predict action using expert agent
            action, _ = agent.predict(obs, deterministic=False)
            action = int(action)
            original_action = action
            perturbed = False

            # apply perturbation
            if perturbation and random.random() < perturbation_level:
                all_actions = list(range(env.action_space.n))
                all_actions.remove(action)
                action = random.choice(all_actions)
                perturbed = True
                perturbed_count += 1

            # step environment
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # predict next action (for BVE)
            next_action, _ = agent.predict(new_obs, deterministic=False)
            next_action = int(next_action)

            # store full transition
            transition = (obs, action, reward, new_obs, next_action, done, perturbed, original_action)
            episode_transitions.append(transition)

            obs = new_obs
            pbar.update(1)

        dataset.extend(episode_transitions)
        transition_counter += len(episode_transitions)
        episode_idx += 1

    pbar.close()
    env.close()

    print(f"Final dataset length: {len(dataset)}")
    print(f"Number of perturbed actions: {perturbed_count}")

    os.makedirs(save_path, exist_ok=True)
    save_dataset(dataset, save_path, file_name)
    return dataset

def load_dataset(filename):
    """
    Loads a dataset from a file.

    :param filename: name of the file to load the dataset from
    :return: the loaded dataset
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def print_dataset_samples(name, dataset, num_samples=3):
    """
    Prints out a specified number of samples from any dataset.

    :param dataset: dataset from which to print samples
    :param num_samples: number of samples to print (Default is 3)
    """
    print(f"Sample entries from the generated {name} Dataset:")
    print(f"Number of entries: {len(dataset)}")
    for i, sample in enumerate(dataset[:num_samples]):
        print(f"Entry {i}: {sample}")
    
def preprocess_and_split(data, seed, validation_size=0.1):
    """
    Preprocesses the dataset by grouping transitions into full episodes and splitting into training and validation sets.

    :param data: List of transitions, where each transition is a tuple of the form:
        (obs, action, reward, next_obs, next_action, done, perturbed_flag, original_action)
    :param seed: Seed for reproducibility
    :param validation_size: Fraction of data to use for validation (Default is 0.1)
    :return: Tuple of training and validation datasets, each as a list of transitions
    """
    # step 1: Group transitions into full episodes
    episodes = []
    current_episode = []

    for transition in data:
        current_episode.append(transition)
        if transition[5]:  # done flag
            episodes.append(current_episode)
            current_episode = []

    # handle case where last episode is incomplete (no 'done')
    if current_episode:
        episodes.append(current_episode)

    # step 2: Split by episode
    train_eps, val_eps = train_test_split(episodes, test_size=validation_size, random_state=seed)

    # step 3: Flatten episodes back into transition lists
    train_data = list(itertools.chain.from_iterable(train_eps))
    val_data = list(itertools.chain.from_iterable(val_eps))

    return train_data, val_data

def create_dataloaders(train_data, validation_data, batch_size=64, seed=42):
    """
    Creates DataLoader objects for training and validation datasets using the custom Dataset class.

    :param train_data: training data list
    :param validation_data: validation data list
    :param batch_size: batch size for the DataLoader (Default is 64)
    :param seed: seed for random state to ensure reproducibility
    :return: tuple containing training and validation DataLoaders
    """
    train_dataset = SeaquestDataset(train_data)
    validation_dataset = SeaquestDataset(validation_data)

    worker_init = partial(worker_init_fn, seed=seed)
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init
    )

    return train_loader, validation_loader

def analyze_action_distribution(dataset):
    """
    Analyzes the distribution of actions in the dataset.

    :param dataset: List of transitions.
    :return: Dictionary with action counts.
    """
    actions = [transition[1] for transition in dataset]
    unique_actions, counts = np.unique(actions, return_counts=True)
    action_distribution = dict(zip(unique_actions, counts))
    print("Action distribution in expert dataset:", action_distribution)
    return action_distribution

def validate_dataset(path, verbose=True):
    """
    Validates a dataset generated with the updated generate_dataset function.
    Expects 8-tuple format: (obs, action, reward, next_obs, next_action, done, perturbed_flag, original_action)

    :param path: Path to the .pkl dataset file
    :param verbose: Whether to print full results
    :return: (is_valid, stats_dict)
    """
    try:
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return False, {}

    is_valid = True
    total = len(dataset)

    if len(dataset[0]) != 8:
        print("Error: Expected 8-tuple format (obs, action, reward, next_obs, next_action, done, perturbed, original_action)")
        return False, {}

    perturbed_count = 0
    wrong_perturbs = 0
    diagonal_hits = 0
    action_space_size = 18  # seaquest

    for t in dataset:
        obs, action, reward, next_obs, next_action, done, perturbed, original_action = t

        if perturbed:
            perturbed_count += 1
            if action == original_action:
                diagonal_hits += 1
                wrong_perturbs += 1
        else:
            if action != original_action:
                wrong_perturbs += 1

    if verbose:
        print(f"{os.path.basename(path)}")
        print(f"    Total transitions      : {total}")
        print(f"    Perturbed actions      : {perturbed_count}")
        print(f"    Wrong perturbations    : {wrong_perturbs}")
        print(f"    Diagonal (same action) : {diagonal_hits}")
        print(f"    Perturbation rate      : {perturbed_count / total:.2%}")

    if wrong_perturbs > 0:
        print("Found invalid perturbations (e.g., perturbed=True but action == original_action)")
        is_valid = False
    else:
        print("Perturbation logic valid.")

    # episode statistics
    rewards = []
    lengths = []
    current_reward = 0
    current_length = 0

    for t in dataset:
        _, _, reward, _, _, done, _, _ = t
        current_reward += reward
        current_length += 1

        if done:
            rewards.append(current_reward)
            lengths.append(current_length)
            current_reward = 0
            current_length = 0

    if current_length > 0:
        rewards.append(current_reward)
        lengths.append(current_length)

    result = {
        "total_transitions": total,
        "num_episodes": len(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_reward": sum(rewards) / len(rewards),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "cut_off_last_episode": not dataset[-1][5],  # done field
        "last_episode_length": current_length if current_length > 0 else None
    }

    if verbose:
        print(f"\nEpisode Analysis for {os.path.basename(path)}")
        print(f"    ➤ Action space size    : {action_space_size}")
        print(f"    ➤ Number of episodes   : {result['num_episodes']}")
        print(f"    ➤ Reward range         : {result['min_reward']} to {result['max_reward']}")
        print(f"    ➤ Mean reward          : {result['mean_reward']:.2f}")
        print(f"    ➤ Episode Length range : {result['min_length']} to {result['max_length']}")
        print(f"    ➤ Mean episode length  : {result['mean_length']:.2f}")
        print(f"    ➤ Last episode cut?    : {'Yes' if result['cut_off_last_episode'] else 'No'}")
        print(f"    ➤ Last episode length  : {result['last_episode_length'] if result['cut_off_last_episode'] else 'N/A'}")

    return is_valid, result

def load_and_prepare_dataset(dataset_path, batch_size=64, seed=0, validation_size=0.1, dataloaders_dict=None):
    """
    Loads a dataset, splits into train/validation sets, and creates dataloaders.

    :param dataset_path: Path to the .pkl dataset file
    :param batch_size: Batch size for dataloaders
    :param seed: Random seed for splitting
    :param validation_size: Fraction of data for validation
    :param dataloaders_dict: Optional dictionary to store the loaders into
    :return: Dictionary of dataloaders { 'train', 'validation' }
    """
    if dataloaders_dict is None:
        dataloaders_dict = {}

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    print(f"=== Loading {dataset_name} dataset ===")

    data = load_dataset(dataset_path)
    print(f"Preprocessing and splitting {dataset_name} dataset...")

    train_data, validation_data = preprocess_and_split(
        data=data,
        seed=seed,
        validation_size=validation_size
    )

    print(f"Creating dataloaders for {dataset_name}...")

    train_loader, validation_loader = create_dataloaders(
        train_data,
        validation_data,
        batch_size=batch_size,
        seed=seed
    )

    dataloaders_dict[dataset_name] = {
        'train': train_loader,
        'validation': validation_loader
    }

    del data, train_data, validation_data
    gc.collect()

    print(f"Dataloaders ready for: {dataset_name}")
    return dataloaders_dict
