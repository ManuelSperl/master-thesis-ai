
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from IPython.display import display
import matplotlib.patches as mpatches


def plot_action_distribution(dataset_dict):
    """
    Plots dot plots of action distributions for each dataset type and perturbation level.
    
    :param dataset_dict: dict with structure {dataset_name: dataset_list}
    """
    plt.figure(figsize=(16, 8))
    for idx, (dataset_name, dataset) in enumerate(dataset_dict.items()):
        plt.subplot(1, len(dataset_dict), idx + 1)
        actions = [sample[1] for sample in dataset]
        unique, counts = np.unique(actions, return_counts=True)
        plt.scatter(unique, counts, s=80, label=dataset_name)
        plt.title(f"Action Distribution\n{dataset_name}")
        plt.xlabel("Action ID")
        plt.ylabel("Count")
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_perturbation_effects(datasets_by_perturbation):
    """
    Plots perturbation % vs number of unique actions or entropy.
    
    :param datasets_by_perturbation: dict with keys as perturbation % and values as datasets
    """
    perturb_levels = []
    unique_action_counts = []

    for perturb_level, dataset in datasets_by_perturbation.items():
        actions = [sample[1] for sample in dataset]
        unique_action_counts.append(len(set(actions)))
        perturb_levels.append(float(perturb_level))

    plt.figure(figsize=(8, 6))
    plt.plot(perturb_levels, unique_action_counts, marker='o')
    plt.title("Perturbation % vs Unique Actions")
    plt.xlabel("Perturbation Level (%)")
    plt.ylabel("Unique Actions")
    plt.grid(True)
    plt.show()


def visualize_perturbation_samples(original_dataset, perturbed_dataset, num_samples=3):
    """
    Show side-by-side comparison of original vs perturbed images.
    """
    for i in range(num_samples):
        orig_obs = original_dataset[i][0]
        pert_obs = perturbed_dataset[i][0]

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(orig_obs)
        axs[0].set_title("Original Observation")
        axs[1].imshow(pert_obs)
        axs[1].set_title("Perturbed Observation")

        for ax in axs:
            ax.axis('off')
        plt.suptitle(f"Sample {i + 1}")
        plt.tight_layout()
        plt.show()


def plot_reward_distribution(reward_dict):
    """
    Plot reward distribution across datasets (if available).
    
    :param reward_dict: dict {dataset_name: list_of_rewards}
    """
    plt.figure(figsize=(12, 6))
    for name, rewards in reward_dict.items():
        sns.kdeplot(rewards, label=name, fill=True)

    plt.title("Reward Distribution per Dataset")
    plt.xlabel("Reward")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
    