# dataset_utils.py

import pickle
import numpy as np
import random
from tqdm import tqdm
import torch
import os
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


def generate_dataset_until_n(env, model, target_size=150_000, perturbation=False, perturbation_level=0.05, save_path='', file_name='', plotting=False):
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
    :param plotting: If True, include original and perturbed actions + flag
    :return: The full list of transitions
    """
    tqdm.write(f"Generating dataset with {target_size} transitions...")

    dataset = []
    perturbed_count = 0

    pbar = tqdm(total=target_size, desc="Collecting transitions")

    while len(dataset) < target_size:
        obs, _ = env.reset()
        done = False

        while not done and len(dataset) < target_size:
            # Predict action using expert model
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            original_action = action
            perturbed = False

            # Apply perturbation (only if different)
            if perturbation and random.random() < perturbation_level:
                all_actions = list(range(env.action_space.n))
                all_actions.remove(action)
                action = random.choice(all_actions)
                perturbed = True
                perturbed_count += 1

            # Take a step
            new_obs, reward, done = env.step(action)[:3]

            # Append to dataset
            if plotting:
                dataset.append((obs, original_action, action, reward, new_obs, done, perturbed))
            else:
                dataset.append((obs, action, reward, new_obs, done))

            obs = new_obs
            pbar.update(1)

    pbar.close()
    env.close()

    print(f"Final dataset length: {len(dataset)}")
    print(f"Number of perturbed actions: {perturbed_count}")

    # Adjust filename if needed
    if plotting:
        file_name = file_name.replace('.pkl', '_plotting.pkl')

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
    

def check_dataset(check, env, num_samples=3, dataset=None, dataset_perturbed=None):
    """
    ...

    :param check: ...
    :param dataset: ...
    :param num_samples: ...
    """
    
    # 1 - Check the Dataset Length
    if check == 1:
        print(f"Total transitions collected: {len(dataset)}")
    
    # 2 - Inspect Sample Transitions
    elif check == 2:
        # Print the first few entries of the dataset to check their structure
        for i in range(num_samples):
            obs, action, reward, new_obs, done = dataset[i]
            print(f"Transition {i+1}:")
            print(f"  Observation shape: {obs.shape}")
            print(f"  Action: {action}")
            print(f"  Reward: {reward}")
            print(f"  New Observation shape: {new_obs.shape}")
            print(f"  Done flag: {done}")
            print("-" * 30)
    
    # 3 - Visualize Observations
    elif check == 3:
        # Visualize the first observation
        plt.imshow(dataset[0][0])
        plt.title("First Observation")
        plt.axis('off')
        plt.show()
    
    # 4 - Validate Actions Against the Action Space
    elif check == 4:
        valid_actions = list(range(env.action_space.n))
        invalid_actions = [action for _, action, _, _, _ in dataset if action not in valid_actions]

        if invalid_actions:
            print(f"Found invalid actions: {invalid_actions}")
        else:
            print("All actions are valid.")
    
    # 5 - Analyze Reward Distribution
    elif check == 5:
        rewards = [reward for _, _, reward, _, _ in dataset]
        print(f"Reward Statistics:")
        print(f"  Total rewards collected: {sum(rewards)}")
        print(f"  Average reward per transition: {np.mean(rewards)}")
        print(f"  Min reward: {np.min(rewards)}")
        print(f"  Max reward: {np.max(rewards)}")
    
    # 6 - Check Done Flags and Episode Termination
    elif check == 6:
        done_flags = [done for _, _, _, _, done in dataset]
        num_episodes_recorded = done_flags.count(True)
        print(f"Number of episodes recorded: {num_episodes_recorded}")
    
    # 7 - Verify Sequential Consistency
    elif check == 7:
        for i in range(len(dataset) - 1):
            obs_current = dataset[i][3]  # new_obs of current
            obs_next = dataset[i + 1][0]  # obs of next
            done = dataset[i][4]
            
            if not done:
                if not np.array_equal(obs_current, obs_next):
                    print(f"Inconsistency found between transitions {i} and {i+1}")
                    break
        else:
            print("All sequential observations are consistent.")
    
    # 8 - Validate Data Types
    elif check == 8:
        # Check data types of the first transition
        obs_dtype = dataset[0][0].dtype
        action_dtype = type(dataset[0][1])
        reward_dtype = type(dataset[0][2])
        done_dtype = type(dataset[0][4])

        print(f"Observation dtype: {obs_dtype}")
        print(f"Action dtype: {action_dtype}")
        print(f"Reward dtype: {reward_dtype}")
        print(f"Done dtype: {done_dtype}")
    
    # 9 - Check Perturbations (If Applied)
    elif check == 9:
        # Assuming you have both datasets
        for i in range(num_samples):
            obs_orig = dataset[i][0]
            obs_perturbed = dataset_perturbed[i][0]
            difference = np.mean(np.abs(obs_orig.astype(np.float32) - obs_perturbed.astype(np.float32)))
            print(f"Average difference in observations at transition {i+1}: {difference}")

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

        print(f"States Batch Data Type: {states.dtype}")      # Expected: torch.float32
        print(f"Actions Batch Data Type: {actions.dtype}")    # Expected: torch.long
        print(f"Rewards Batch Data Type: {rewards.dtype}")    # Expected: torch.float32
        print(f"Next States Batch Data Type: {next_states.dtype}")  # Expected: torch.float32
        print(f"Dones Batch Data Type: {dones.dtype}")        # Expected: torch.float32 or torch.bool

        # Display first sample's information
        print("\nFirst Sample Details:")
        print(f"First State Shape: {states[0].shape}")        # Expected: (channels, height, width)
        print(f"First State Min/Max: {states[0].min().item():.4f}/{states[0].max().item():.4f}")  # Should be between 0 and 1
        print(f"First Action: {actions[0].item()}")
        print(f"First Reward: {rewards[0].item():.4f}")
        print(f"First Next State Shape: {next_states[0].shape}")  # Expected: (channels, height, width)
        print(f"First Done: {dones[0].item()}")

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