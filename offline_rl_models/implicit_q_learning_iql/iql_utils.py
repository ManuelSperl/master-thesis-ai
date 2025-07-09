# iql_utils.py 

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time
import numpy as np
import gc
import pandas as pd
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import importlib

import auxiliary_methods.tb_logger
importlib.reload(auxiliary_methods.tb_logger)
from auxiliary_methods.tb_logger import TensorBoardLogger

import datasets.dataset_utils
importlib.reload(datasets.dataset_utils)

from datasets.dataset_utils import set_all_seeds

import auxiliary_methods.utils
importlib.reload(auxiliary_methods.utils)
from auxiliary_methods.utils import create_env, save_return_stats, free_up_memory

import offline_rl_models.implicit_q_learning_iql.iql_model
importlib.reload(offline_rl_models.implicit_q_learning_iql.iql_model)
from offline_rl_models.implicit_q_learning_iql.iql_model import IQLModel

def evaluate_IQL_reward(env, agent, n_episodes, device, max_steps_per_episode=1000, debug=False):
    """
    Evaluate the IQL agent on the given environment.

    :param env: The environment to evaluate the agent on.
    :param agent: The IQL agent to evaluate.
    :param n_episodes: Number of episodes to run for evaluation.
    :param device: Device to run the evaluation on (CPU or GPU).
    :param max_steps_per_episode: Maximum number of steps per episode.
    :param debug: If True, print debug information.
    :return: List of total rewards for each episode.
    """
    rewards = []
    agent.actor.eval()

    for episode in range(n_episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                state = torch.tensor(observation, dtype=torch.float32)
                state = state.permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                
                action = agent.get_action(state, eval=True)
                if isinstance(action, torch.Tensor):
                    action = action.item()


            observation, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)

        if debug:
            print(f"Finished Episode {episode+1}: Total Reward = {total_reward}, Steps Taken = {steps}")

        free_up_memory([state, action, observation, reward, terminated, truncated])

    return rewards

def train_and_evaluate_IQL(dataloaders, epochs, seeds, dataset, env_id, seed, device):
    """
    Train and evaluate the Implicit Q-Learning (IQL) model on the specified dataset.

    :param dataloaders: Dictionary containing training and validation data loaders.
    :param epochs: Number of training epochs.
    :param seeds: Number of random seeds to use for training.
    :param dataset: Name of the dataset to train on.
    :param env_id: Environment ID for the IQL agent.
    :param seed: Global seed for reproducibility.
    :param device: Device to run the training on (CPU or GPU).
    """
    logdir = 'offline_rl_models/implicit_q_learning_iql/iql_logs'
    stats_to_save = {}

    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training IQL on {dataset_name}")

        # extract metadata
        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()  # e.g., 'expert_dataset'
        perturbation_level = parts[-1]  # e.g., '20'

        # init logs
        all_actor_losses, all_critic1_losses, all_critic2_losses, all_value_losses = [], [], [], []
        all_val_losses, all_rewards = [], []
        all_actor_steps, all_critic1_steps, all_critic2_steps, all_value_steps = [], [], [], []
        all_val_steps, all_reward_steps = [], []
        q_values_by_seed = {}

        # loop through seeds
        for seed_idx in range(seeds):
            print(f"-- Starting Seed {seed_idx + 1}/{seeds} --")

            # ---- Setup Phase ----
            iql_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)
            iql_env, iql_action_dim = create_env(iql_logger, seed, env_id, True, seed_number=f'{seed_idx+1}')
            iql_agent = IQLModel(iql_action_dim, device, seed, seed_idx)
            criterion = nn.CrossEntropyLoss()

            main_tag = f'{dataset_name}_Seed_{seed_idx + 1}'

            actor_losses, critic1_losses, critic2_losses, value_losses = [], [], [], []
            val_losses, rewards = [], []
            actor_steps, critic1_steps, critic2_steps, value_steps = [], [], [], []
            val_steps, reward_steps = [], []
            q_values_by_seed[seed_idx] = {'avg_pred_q_values': {}, 'avg_target_q_values': {}}
            actor_step = critic1_step = critic2_step = value_step = val_step = reward_step = 0

            # start Offline Training -> loop through epochs
            for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
                # ---- Train Phase ----
                iql_agent.train()
                for states, actions, rews, next_states, _, dones, *_ in loaders['train']:
                    states = states.to(device)
                    actions = actions.to(device)
                    rews = rews.unsqueeze(1).to(device)
                    next_states = next_states.to(device)
                    dones = dones.unsqueeze(1).to(device)

                    out = iql_agent.learn((states, actions, rews, next_states, dones))
                    a_loss, c1_loss, c1_q, c1_tgt, c2_loss, c2_q, c2_tgt, v_loss = out

                    actor_losses.append(a_loss)
                    actor_steps.append(actor_step)
                    iql_logger.log(f'{main_tag}/Actor Loss', a_loss, actor_step)
                    actor_step += 1

                    critic1_losses.append(c1_loss)
                    critic1_steps.append(critic1_step)
                    iql_logger.log(f'{main_tag}/Critic 1 Loss', c1_loss, critic1_step)
                    critic1_step += 1

                    critic2_losses.append(c2_loss)
                    critic2_steps.append(critic2_step)
                    iql_logger.log(f'{main_tag}/Critic 2 Loss', c2_loss, critic2_step)
                    critic2_step += 1

                    value_losses.append(v_loss)
                    value_steps.append(value_step)
                    iql_logger.log(f'{main_tag}/Value Loss', v_loss, value_step)
                    value_step += 1

                    # log Q-values with a shared step (e.g., critic1_step)
                    q_step = critic1_step
                    q_values_by_seed[seed_idx]['avg_pred_q_values'][q_step] = [(c1_q.mean().item() + c2_q.mean().item()) / 2]
                    q_values_by_seed[seed_idx]['avg_target_q_values'][q_step] = [(c1_tgt.mean().item() + c2_tgt.mean().item()) / 2]

                    free_up_memory([states, actions, rews, next_states, dones])

                # ---- Validation Phase ----
                iql_agent.actor.eval()
                epoch_val_losses = []
                with torch.no_grad():
                    for states, actions, *_ in loaders['validation']:
                        states, actions = states.to(device), actions.to(device)
                        logits = iql_agent.actor(states)
                        loss = criterion(logits, actions)

                        val_losses.append(loss.item())
                        epoch_val_losses.append(loss.item())
                        val_steps.append(val_step)
                        iql_logger.log(f'{main_tag}/Validation Loss', loss.item(), val_step)
                        val_step += 1

                        free_up_memory([states, actions, logits, loss])

                if epoch_val_losses:
                    for opt in [iql_agent.actor_optimizer, iql_agent.critic1_optimizer, iql_agent.critic2_optimizer, iql_agent.value_optimizer]:
                        for pg in opt.param_groups:
                            pg['lr'] *= 0.98 if np.mean(epoch_val_losses) > 0.5 else 1.0

                # ---- Reward Phase ----
                reward_list = evaluate_IQL_reward(iql_env, iql_agent, n_episodes=15, device=device, max_steps_per_episode=3000)
                for r in reward_list:
                    rewards.append(r)
                    reward_steps.append(reward_step)
                    iql_logger.log(f'{main_tag}/Reward', r, reward_step)
                    reward_step += 1

            # store all seed data
            all_actor_losses.append(actor_losses)
            all_critic1_losses.append(critic1_losses)
            all_critic2_losses.append(critic2_losses)
            all_value_losses.append(value_losses)
            all_val_losses.append(val_losses)
            all_rewards.append(rewards)
            all_actor_steps.append(actor_steps)
            all_critic1_steps.append(critic1_steps)
            all_critic2_steps.append(critic2_steps)
            all_value_steps.append(value_steps)
            all_val_steps.append(val_steps)
            all_reward_steps.append(reward_steps)

            print(f"Finished Training on {dataset_name}")
            print(f"    ➤ Avg Actor Loss: {np.mean(actor_losses):.5f}")
            print(f"    ➤ Avg Critic1 Loss: {np.mean(critic1_losses):.5f}")
            print(f"    ➤ Avg Critic2 Loss: {np.mean(critic2_losses):.5f}")
            print(f"    ➤ Avg Value Loss: {np.mean(value_losses):.5f}")
            print(f"    ➤ Avg Validation Loss: {np.mean(val_losses):.5f}")
            print(f"    ➤ Avg Reward: {np.mean(rewards):.2f}")

            # save the model
            print("Saving the model...")
            model_path = f"{logdir}/{dataset_type}/{perturbation_level}/iql_model_{perturbation_level}_seed{seed_idx + 1}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(iql_agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            iql_env.close()
            time.sleep(1.5)

        stats_to_save = {
            'actor_losses': all_actor_losses,
            'critic1_losses': all_critic1_losses,
            'critic2_losses': all_critic2_losses,
            'value_losses': all_value_losses,
            'val_losses': all_val_losses,
            'rewards': all_rewards,
            'actor_steps': all_actor_steps,
            'critic1_steps': all_critic1_steps,
            'critic2_steps': all_critic2_steps,
            'value_steps': all_value_steps,
            'val_steps': all_val_steps,
            'reward_steps': all_reward_steps,
            'q_values': q_values_by_seed,
            'dataset_name': dataset_name,
            'num_seeds': seeds
        }

    stats_path = f"{logdir}/{dataset_type}/{perturbation_level}/stats_{perturbation_level}.pkl"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    save_return_stats(stats_to_save, stats_path)
    print(f"Return Stats saved to {stats_path}")

    iql_logger.close()