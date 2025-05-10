
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss_curves(loss_data, model_name):
    """
    Plots training loss curves across trials.

    :param loss_data: list of lists (one per trial)
    :param model_name: name of the model (BC/DQN/IQL)
    """
    avg_loss = np.mean(loss_data, axis=0)
    std_loss = np.std(loss_data, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_loss, label="Average Loss")
    plt.fill_between(range(len(avg_loss)), avg_loss - std_loss, avg_loss + std_loss, alpha=0.3)
    plt.title(f"{model_name} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_test_loss_curves(test_loss_data, model_name):
    avg_loss = np.mean(test_loss_data, axis=0)
    std_loss = np.std(test_loss_data, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_loss, label="Average Test Loss")
    plt.fill_between(range(len(avg_loss)), avg_loss - std_loss, avg_loss + std_loss, alpha=0.3)
    plt.title(f"{model_name} - Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_reward_curves(reward_data, model_name):
    avg_reward = np.mean(reward_data, axis=0)
    std_reward = np.std(reward_data, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_reward, label="Average Reward")
    plt.fill_between(range(len(avg_reward)), avg_reward - std_reward, avg_reward + std_reward, alpha=0.3)
    plt.title(f"{model_name} - Reward per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_q_values_dynamics(q_values_dict, trial_idx=0):
    """
    Plots Q-value predictions and targets over epochs for IQL.

    :param q_values_dict: Dict with structure: q_values[trial]['avg_pred_q_values'][epoch] -> list of values
    :param trial_idx: trial number to visualize
    """
    epochs = sorted(q_values_dict[trial_idx]['avg_pred_q_values'].keys())
    pred_q = [np.mean(q_values_dict[trial_idx]['avg_pred_q_values'][e]) for e in epochs]
    target_q = [np.mean(q_values_dict[trial_idx]['avg_target_q_values'][e]) for e in epochs]

    plt.figure(figsize=(10, 5))
    plt.plot(pred_q, label="Predicted Q", marker='o')
    plt.plot(target_q, label="Target Q", marker='x')
    plt.title(f"IQL - Q-Value Dynamics (Trial {trial_idx})")
    plt.xlabel("Epoch")
    plt.ylabel("Q-Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    