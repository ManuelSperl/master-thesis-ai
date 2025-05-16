# dataset_utils.py

import pickle
import numpy as np
import random
from tqdm import tqdm
import torch
import os
import gc
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from functools import partial  # Import partial for passing arguments to worker_init_fn

import importlib
import datasets.seaquest_dataset
importlib.reload(datasets.seaquest_dataset)

# Import SeaquestDataset from a separate module
from datasets.seaquest_dataset import SeaquestDataset

def evaluate_dqn_agent(env, model, num_episodes=5):
    episode_rewards = []
    action_counts = {}
    # use tqdm
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        total_rewards = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # Convert action to scalar if it's a NumPy array
            if isinstance(action, np.ndarray):
                action = action.item()
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_rewards += reward
            # Log action
            action_counts[action] = action_counts.get(action, 0) + 1
            
        episode_rewards.append(total_rewards)
        
    average_reward = np.mean(episode_rewards)
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")
    print(f"Action Counts: {action_counts}")
    return average_reward, action_counts

# Define worker_init_fn at the top level
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
    Creates a Gym environment with specified wrappers and configurations.

    :param env_id: Identifier for the Gym environment.
    :param seed: Seed value for reproducibility.
    :param frame_stack: Number of frames to stack (default is 4).

    :return: An instance of the configured Gym environment.
    """
    # Create the base environment
    env = gym.make(env_id)

    # Seed all parts of the environment to ensure reproducibility
    if seed is not None:
        # Access the unwrapped environment directly for setting the seed
        unwrapped_env = env.unwrapped
        unwrapped_env.seed(seed)
        unwrapped_env.action_space.seed(seed)
        unwrapped_env.observation_space.seed(seed)
    
    #print(f"Environment '{env_id}' created with seed {seed}")
    #print(f"Action Space: {env.action_space}, Number of Actions: {env.action_space.n}")
    #print(f"Observation Space: {env.observation_space}")

    return env

def set_all_seeds(seed):
    """
    Function that sets a seed for reproducibility.

    :param seed: the seed to set
    """
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU seed
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  # Avoids non-deterministic optimizations

    set_random_seed(seed)  # For stable_baselines3


def save_dataset(dataset, path, filename):
    """
    Saves a dataset to a specified path and file.

    :param dataset: dataset to save
    :param path: directory where the file will be saved
    :param filename: name of the file to save the dataset to
    """
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    # Construct the full file path
    full_path = os.path.join(path, filename)
    
    # Save the dataset to the specified file
    with open(full_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {full_path}")


def generate_dataset(env, model, target_size=150_000, perturbation=False, perturbation_level=0.05, save_path='', file_name='', for_stats=False):
    """
    Generates a dataset with a fixed number of transitions (target_size) using an expert policy.
    Optionally perturbs actions to simulate suboptimality.

    :param env: Gym or VecEnv environment object (Seaquest)
    :param model: Trained expert model (e.g., DQN, PPO)
    :param target_size: Total number of transitions to collect
    :param perturbation: Whether to apply random action perturbation
    :param perturbation_level: Proportion of transitions to perturb (e.g., 0.05 for 5%)
    :param save_path: Directory to save the dataset
    :param file_name: File name for the saved dataset
    :param for_stats: If True, include original and perturbed actions + flag
    :return: The full list of transitions
    """
    tqdm.write(f"Generating dataset with at least {target_size} transitions (full episodes only)...")

    dataset = []
    perturbed_count = 0
    transition_counter = 0

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
            # Predict action using expert model
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            original_action = action
            perturbed = False

            # Apply perturbation
            if perturbation and random.random() < perturbation_level:
                all_actions = list(range(env.action_space.n))
                all_actions.remove(action)
                action = random.choice(all_actions)
                perturbed = True
                perturbed_count += 1

            # Step environment
            new_obs, reward, done = env.step(action)[:3]

            if for_stats:
                episode_transitions.append((obs, original_action, action, reward, new_obs, done, perturbed))
            else:
                episode_transitions.append((obs, action, reward, new_obs, done))

            obs = new_obs
            pbar.update(1)

        # Only after episode ends:
        dataset.extend(episode_transitions)
        transition_counter += len(episode_transitions)

    pbar.close()
    env.close()

    print(f"Final dataset length: {len(dataset)}")
    print(f"Number of perturbed actions: {perturbed_count}")

    # Adjust filename if needed
    if for_stats:
        file_name = file_name.replace('.pkl', '_stats.pkl')

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
    
def preprocess_and_split(data, seed, test_size=0.2, tune_size=0.1):
    """
    Splits the dataset into training, tuning, and test sets.

    :param data: dataset to split
    :param seed: seed for random state to ensure reproducibility
    :param test_size: test split size (default 0.2)
    :param tune_size: tuning split size (default 0.1)

    :return: tuple containing training, test, and tuning sets
    """

    # Split the data into temporary training and combined test and tuning sets
    train_data, temp_test_data = train_test_split(
        data, test_size=test_size + tune_size, random_state=seed
    )

    # Calculate relative size of tuning set w.r.t temporary test set
    tune_size_relative = tune_size / (test_size + tune_size)

    # Split temporary test data into test and tuning sets
    test_data, tune_data = train_test_split(
        temp_test_data, test_size=tune_size_relative, random_state=seed
    )

    return train_data, test_data, tune_data

def create_dataloaders(train_data, test_data, tune_data, batch_size=64, seed=42):
    """
    Creates DataLoader objects for training, testing, and tuning datasets using the custom Dataset class.

    :param train_data: training data list
    :param test_data: testing data list
    :param tune_data: tuning data list
    :param batch_size: batch size for the DataLoader (Default is 64)
    :param seed: seed for random state to ensure reproducibility

    :return: tuple containing training, testing, and tuning DataLoaders
    """
    # Create custom datasets
    train_dataset = SeaquestDataset(train_data)
    test_dataset = SeaquestDataset(test_data)
    tune_dataset = SeaquestDataset(tune_data)

    # Create a partial function for worker_init_fn with the seed
    worker_init = partial(worker_init_fn, seed=seed)

    # Set num_workers=0 to avoid pickling issues (alternatively, you can keep num_workers > 0 if using partial)
    num_workers = 0  # Set to 0 to avoid pickling issues on Windows

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init
    )
    tune_loader = DataLoader(
        tune_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init
    )

    return train_loader, test_loader, tune_loader

from itertools import islice

def inspect_dataset_sample(dataloader, num_samples=1):
    """
    Inspects a sample from the dataset to check the shapes and data types of the data.

    :param dataloader: DataLoader object to inspect
    :param num_samples: Number of samples to inspect (default is 1).
    """
    # Use islice to get only a few samples
    dataiter = islice(dataloader, num_samples)  # Create an iterator from the DataLoader

    for i, (states, actions, rewards, next_states, dones) in enumerate(dataiter):
        print(f"--- Sample {i+1} ---")
        print(f"States Batch Shape: {states.shape}")          # Expected: (batch_size, channels, height, width)
        print(f"Actions Batch Shape: {actions.shape}")        # Expected: (batch_size,)
        print(f"Rewards Batch Shape: {rewards.shape}")        # Expected: (batch_size,)
        print(f"Next States Batch Shape: {next_states.shape}")# Expected: (batch_size, channels, height, width)
        print(f"Dones Batch Shape: {dones.shape}")            # Expected: (batch_size,)

        # Stop after num_samples are inspected
        if i + 1 >= num_samples:
            break

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

def validate_dataset(path, expected_length=None, expect_plotting_format=True, verbose=True):
    """
    Validates a dataset generated with the expert_dataset function.

    :param path: Path to the .pkl dataset file
    :param expected_length: If provided, will warn if dataset length is different
    :param expect_plotting_format: If True, expects (obs, original_action, action, reward, new_obs, done, perturbed)
    :param verbose: Whether to print full results
    :return: True if valid, False if issues found
    """
    try:
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return False

    is_valid = True
    total = len(dataset)

    if expected_length and total != expected_length:
        print(f"Length mismatch: expected {expected_length}, got {total}")
        is_valid = False

    has_perturb_info = expect_plotting_format and len(dataset[0]) == 7
    if expect_plotting_format and not has_perturb_info:
        print("Expected 7-tuple format (with perturbation), but got 5-tuple")
        return False

    perturbed_count = 0
    wrong_perturbs = 0
    diagonal_hits = 0
    action_space_size = 18  # hardcoded for Seaquest

    for t in dataset:
        if has_perturb_info:
            _, original, perturbed, _, _, _, is_perturbed = t
            if is_perturbed:
                perturbed_count += 1
                if original == perturbed:
                    diagonal_hits += 1
                    wrong_perturbs += 1
            else:
                if original != perturbed:
                    wrong_perturbs += 1
        else:
            # nothing to validate in non-plotting datasets
            break

    if verbose:
        print(f"{os.path.basename(path)}")
        print(f"    Total transitions      : {total}")
        print(f"    Perturbed actions      : {perturbed_count}")
        if has_perturb_info:
            print(f"    Wrong perturbations    : {wrong_perturbs}")
            print(f"    Diagonal (same action) : {diagonal_hits}")
            print(f"    Perturbation rate      : {perturbed_count / total:.2%}")

    if wrong_perturbs > 0:
        print("Found invalid perturbations (e.g., perturbed=True but original == action)")
        is_valid = False
    else:
        print("Perturbation logic valid.")

    rewards = []
    lengths = []
    current_reward = 0
    current_length = 0

    for transition in dataset:
        if expect_plotting_format and len(transition) == 7:
            _, _, _, reward, _, done, _ = transition
        else:
            _, _, reward, _, done = transition

        current_reward += reward
        current_length += 1

        if done:
            rewards.append(current_reward)
            lengths.append(current_length)
            current_reward = 0
            current_length = 0

    # Append last episode if it wasn't terminated properly (cut off)
    if current_length > 0:
        rewards.append(current_reward)
        lengths.append(current_length)

    result = {
        "total_transitions": len(dataset),
        "num_episodes": len(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_reward": sum(rewards) / len(rewards),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "cut_off_last_episode": not dataset[-1][-2],  # done=False
        "last_episode_length": current_length if current_length > 0 else None
    }

    if verbose:
        print(f"Episode Analysis for {os.path.basename(path)}")
        print(f"    ➤ Action space size    : {action_space_size}")
        print(f"    ➤ Number of episodes   : {result['num_episodes']}")
        print(f"    ➤ Reward range         : {result['min_reward']} to {result['max_reward']}")
        print(f"    ➤ Mean reward          : {result['mean_reward']:.2f}")
        print(f"    ➤ Episode Length range : {result['min_length']} to {result['max_length']}")
        print(f"    ➤ Mean episode length  : {result['mean_length']:.2f}")
        print(f"    ➤ Last episode cut?    : {'Yes' if result['cut_off_last_episode'] else 'No'}")
        print(f"    ➤ Last episode length  : {result['last_episode_length'] if result['cut_off_last_episode'] else 'N/A'}")


    return is_valid, result


def load_and_prepare_dataset(dataset_path, batch_size=64, seed=0, test_size=0.2, tune_size=0.1, dataloaders_dict=None):
    """
    Loads a dataset, preprocesses it, splits into train/test/tune sets,
    creates dataloaders, and stores them in the provided dictionary.

    :param dataset_path: Path to the .pkl dataset file
    :param batch_size: Batch size for dataloaders
    :param seed: Random seed for splitting
    :param test_size: Fraction of data for test set
    :param tune_size: Fraction of training data for tuning
    :param dataloaders_dict: Optional dictionary to store the loaders into
    :return: Dictionary of dataloaders { 'train', 'test', 'tuning' }
    """
    if dataloaders_dict is None:
        dataloaders_dict = {}

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    print(f"=== Loading {dataset_name} dataset ===")

    # Load dataset
    data = load_dataset(dataset_path)

    print(f"Preprocessing and splitting {dataset_name} dataset...")

    # Preprocess and split
    train_data, test_data, tune_data = preprocess_and_split(
        data=data,
        seed=seed,
        test_size=test_size,
        tune_size=tune_size
    )

    print(f"Creating dataloaders for {dataset_name}...")

    train_loader, test_loader, tune_loader = create_dataloaders(
        train_data,
        test_data,
        tune_data,
        batch_size=batch_size,
        seed=seed
    )

    dataloaders_dict[dataset_name] = {
        'train': train_loader,
        'test': test_loader,
        'tuning': tune_loader
    }

    # Clear variables
    del data, train_data, test_data, tune_data
    gc.collect()

    print(f"✅ Dataloaders ready for: {dataset_name}")
    return dataloaders_dict
