# iql_utils.py 

# ----- PyTorch imports -----
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time

# ----- Python imports -----
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
from auxiliary_methods.utils import create_env, save_return_stats

import implicit_q_learning_iql.iql_model
importlib.reload(implicit_q_learning_iql.iql_model)
from implicit_q_learning_iql.iql_model import IQLModel

def adjust_learning_rate(agent, decay_factor):
    """
    Adjusts the learning rate of all optimizers within an agent by a specified decay factor.
    """
    for optimizer in [
        agent.actor_optimizer, 
        agent.critic1_optimizer, 
        agent.critic2_optimizer, 
        agent.value_optimizer
    ]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor


def train_one_IQL_epoch(agent, loader):
    """
    Conducts one epoch of training for an IQL agent using the provided data loader.
    """
    agent.train()

    # lists to store losses and Q-values
    actor_losses, critic1_losses, critic2_losses, value_losses = [], [], [], []
    avg_pred_q_values, avg_target_q_values = [], []

    if len(loader) == 0:
        return 0.0  # ensure we don't divide by zero

    for states, actions, rewards, next_states, dones in loader:
        states, actions = states.to(agent.device), actions.to(agent.device)
        rewards = rewards.unsqueeze(1).to(agent.device)
        next_states, dones = next_states.to(agent.device), dones.unsqueeze(1).to(agent.device)
    
        actor_loss, critic1_loss, critic1_pred_q, critic1_target_q, critic2_loss, critic2_pred_q, critic2_target_q, value_loss = agent.learn((states, actions, rewards, next_states, dones))

        actor_losses.append(actor_loss)
        critic1_losses.append(critic1_loss)
        critic2_losses.append(critic2_loss)
        value_losses.append(value_loss)
        avg_pred_q_values.append((critic1_pred_q.mean().item() + critic2_pred_q.mean().item()) / 2)
        avg_target_q_values.append((critic1_target_q.mean().item() + critic2_target_q.mean().item()) / 2)

        # free memory after each batch
        del states, actions, rewards, next_states, dones, actor_loss, critic1_loss, critic1_pred_q, critic1_target_q, critic2_loss, critic2_pred_q, critic2_target_q, value_loss
        torch.cuda.empty_cache()
        gc.collect()

    return np.mean(actor_losses), np.mean(critic1_losses), np.mean(critic2_losses), np.mean(value_losses), avg_pred_q_values, avg_target_q_values


def evaluate_IQL_loss(agent, criterion, data_loader, device):
    """
    Evaluates the IQL model on a dataset (test loss).
    """
    agent.actor.eval()
    total_loss = 0.0
    total_samples = 0

    # compute evaluation Loss from dataset
    with torch.no_grad():
        for states, actions, _, _, _ in data_loader:
            states, actions = states.to(device), actions.to(device).long()

            # forward pass
            logits = agent.actor(states)
            loss = criterion(logits, actions)

            total_loss += loss.item() * states.size(0)
            total_samples += states.size(0)

            # free memory after each episode
            del states, actions, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

    return total_loss / total_samples if total_samples > 0 else 0.0

def evaluate_IQL_losses(agent, data_loader, device):
    """
    Evaluate the IQL model on a test set, computing:
      - Actor classification loss (CrossEntropy)
      - Critic TD loss (MSE or Huber)
      - Value loss (expectile regression)
    """
    agent.eval()

    # Define losses
    actor_criterion = nn.CrossEntropyLoss()
    td_criterion = nn.SmoothL1Loss()  # or MSELoss()

    # Accumulators
    total_actor_loss = 0.0
    total_critic1_loss = 0.0
    total_critic2_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for states, actions, rewards, next_states, dones in data_loader:
            batch_size = states.size(0)
            total_samples += batch_size

            # Move to device
            states = states.to(device)
            actions = actions.to(device).long()
            rewards = rewards.unsqueeze(1).to(device)
            next_states = next_states.to(device)
            dones = dones.unsqueeze(1).to(device)

            # 1️⃣ Actor Loss (CrossEntropy)
            logits = agent.actor(states)
            actor_loss = actor_criterion(logits, actions)

            # 2️⃣ Critic TD Loss
            next_values = agent.value(next_states)
            next_values = torch.clamp(next_values, min=-100, max=100)
            q_target = rewards + (1 - dones) * 0.99 * next_values

            q_target = torch.clamp(q_target, min=-300, max=300)

            q1_pred = agent.critic1(states, actions)
            q2_pred = agent.critic2(states, actions)

            critic1_loss = td_criterion(q1_pred, q_target)
            critic2_loss = td_criterion(q2_pred, q_target)

            # 3️⃣ Value Loss (Expectile regression)
            with torch.no_grad():
                min_q = torch.min(q1_pred, q2_pred)
            v_pred = agent.value(states)

            diff = min_q - v_pred
            expectile = 0.95
            value_loss = ((torch.where(diff > 0, expectile, (1 - expectile)) * diff**2)).mean()
            value_loss = torch.clamp(value_loss, max=1e4)

            # Accumulate
            total_actor_loss += actor_loss.item() * batch_size
            total_critic1_loss += critic1_loss.item() * batch_size
            total_critic2_loss += critic2_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size

            # Clean up
            del states, actions, rewards, next_states, dones
            torch.cuda.empty_cache()
            gc.collect()

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_actor_loss / total_samples,
        total_critic1_loss / total_samples,
        total_critic2_loss / total_samples,
        total_value_loss / total_samples
    )

def evaluate_IQL_reward(env, agent, n_episodes, device, trial_idx, max_steps_per_episode=1000, debug=False):
    ''' Evaluate the IQL agent on the environment for n_episodes '''
    rewards = []

    # make sure environment gets a unique seed per trial
    env.seed(1234 + trial_idx)  
    env.action_space.seed(1234 + trial_idx)

    for episode in range(n_episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0 # step counter

        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                state = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                action = agent.get_action(state, eval=True).squeeze().item()
            
            # handle invalid actions
            if not (0 <= action < env.action_space.n):
                action = max(0, min(action, env.action_space.n - 1))

            # take action in the environment
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1 # increment step counter

            if debug:
                if steps % 500 == 0:
                    print(f"Episode {episode+1}: {steps} steps taken, Total Reward: {total_reward}")

        rewards.append(total_reward)

        if debug:
            print(f"Finished Episode {episode+1}: Total Reward = {total_reward}, Steps Taken = {steps}")

        # free memory after each episode
        del state, action, observation, reward, terminated, truncated
        torch.cuda.empty_cache()
        gc.collect()

    return [np.mean(rewards)]

def train_and_evaluate_IQL(dataloaders, epochs, trials, dataset, env_id, seed, device):
    """
    Trains an Implicit Q-Learning (IQL) agent on the specified dataset.
    """
    logdir = 'implicit_q_learning_iql/iql_logs'
    stats_to_save = {}

    # init dictionary to store Q-values by epoch
    epoch_q_values = {
        trial: {
            'avg_pred_q_values': {epoch: [] for epoch in range(epochs)},
            'avg_target_q_values': {epoch: [] for epoch in range(epochs)}
        } for trial in range(trials)
    }

    # train and evaluate the IQL model on all datasets
    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training IQL on {dataset_name}")

        # split dataset name to extract type and perturbation level
        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()  # e.g., 'expert_dataset'
        perturbation_level = parts[-1]  # e.g., '20'

        # lists to store losses and rewards
        all_actor_losses, all_critic1_losses, all_critic2_losses, all_value_losses, all_test_losses, all_rewards = [], [], [], [], [], []

        # loop through trials
        for trial in range(trials):
            print(f"-- Starting Trial {trial + 1}/{trials} --")

            set_all_seeds(seed + trial) # set all seeds for reproducibility

            # ------- Setup Phase -------
            # logger setup
            iql_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)

            # create the Enviornment
            iql_env, iql_action_dim = create_env(iql_logger, seed + trial, env_id, True, trial_number=f'{trial+1}')

            # initialize the IQL model for each dataset
            iql_agent = IQLModel(iql_action_dim, device, seed + trial, trial)

            criterion = nn.CrossEntropyLoss()

            # lists to store losses and rewards for each trial
            actor_losses, critic1_losses, critic2_losses, value_losses, test_losses, rewards = [], [], [], [], [], []

            # start Offline Training -> loop through epochs
            for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
                # ------- Training Phase -------
                iql_agent.train()
                actor_loss, critic1_loss, critic2_loss, value_loss, avg_pred_q_values, avg_target_q_values = train_one_IQL_epoch(iql_agent, loaders['train'])
                
                # collect losses and Q-values
                actor_losses.append(actor_loss)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
                value_losses.append(value_loss)
                epoch_q_values[trial]['avg_pred_q_values'][epoch] = avg_pred_q_values
                epoch_q_values[trial]['avg_target_q_values'][epoch] = avg_target_q_values

                # log losses
                main_tag = f'{dataset_name}_trial_{trial + 1}'
                iql_logger.log(f'{main_tag}/Actor Loss', actor_loss, epoch + 1)
                iql_logger.log(f'{main_tag}/Critic 1 Loss', critic1_loss, epoch + 1)
                iql_logger.log(f'{main_tag}/Critic 2 Loss', critic2_loss, epoch + 1)
                iql_logger.log(f'{main_tag}/Value Loss', value_loss, epoch + 1)

                # ------- Hyperparameter Tuning Phase -------
                tune_loss = evaluate_IQL_loss(iql_agent, criterion, loaders['tuning'], device)
                iql_logger.log(f'{main_tag}/Tune Loss', tune_loss, epoch + 1)

                # adjust learning rate for all optimizers in the agent
                adjust_learning_rate(iql_agent, 0.98 if tune_loss > 0.5 else 1.0)
                
                # ------- Test Phase -------
                test_loss = evaluate_IQL_loss(iql_agent, criterion, loaders['test'], device)
                test_losses.append(test_loss)
                iql_logger.log(f'{main_tag}/Test Loss', test_loss, epoch + 1)

                # evaluation and reward collection at the end of each epoch
                reward = evaluate_IQL_reward(iql_env, iql_agent, n_episodes=20, device=device, trial_idx=trial, max_steps_per_episode=5000)
                rewards.extend(reward)
                iql_logger.log(f'{main_tag}/Reward', reward[0], epoch + 1) # log reward

                # free Memory After Each Epoch
                torch.cuda.empty_cache()
                gc.collect()
            
            # collect all losses and rewards for all trials
            all_actor_losses.append(actor_losses)
            all_critic1_losses.append(critic1_losses)
            all_critic2_losses.append(critic2_losses)
            all_value_losses.append(value_losses)
            all_test_losses.append(test_losses)
            all_rewards.append(rewards)

            print(f"Finished Training on {dataset_name} - Actor Loss: {actor_loss:.5f}")
            print(f"                                                   - Critic 1 Loss: {critic1_loss:.5f}")
            print(f"                                                   - Critic 2 Loss: {critic2_loss:.5f}")
            print(f"                                                   - Value Loss: {value_loss:.5f}")
            print(f"                                                   - Tuning Loss: {tune_loss:.5f}")
            print(f"                                                   - Test Loss: {test_loss:.5f}")
            print(f"                                                   - Reward: {reward[0]:.5f}")

            # save model after the first trial
            if trial == 0:
                model_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/iql_model_{perturbation_level}.pth"
                torch.save(iql_agent.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")
            
            # close the environment
            iql_env.close()

        stats_to_save = {
            'actor_losses': all_actor_losses,
            'critic1_losses': all_critic1_losses,
            'critic2_losses': all_critic2_losses,
            'value_losses': all_value_losses,
            'test_losses': all_test_losses,
            'rewards': all_rewards,
            'q_values': epoch_q_values,
            'dataset_name': dataset_name,
            'trials': trials
        }

    # save the return stats
    stats_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/stats_{perturbation_level}.pkl"
    save_return_stats(stats_to_save, stats_save_path)
    print(f"Return Stats saved to {stats_save_path}")

    # close the logger
    iql_logger.close()

def continue_training_IQL(dataloaders, further_epochs, dataset, env_id, seed, device):
    """
    Continues training an existing IQL model from a previous trial, including tuning.

    :param dataloaders: Dataset loaders.
    :param further_epochs: Number of additional epochs.
    :param dataset: Dataset name (e.g., 'expert_dataset_perturbation_0').
    :param env_id: Environment ID (e.g., 'Seaquest-v4').
    :param seed: Seed for reproducibility.
    :param device: Training device ('cuda' or 'cpu').
    """
    logdir = 'implicit_q_learning_iql/iql_logs'

    # loop through datasets
    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Continuing IQL Training on {dataset_name}")

        # split dataset name to extract type and perturbation level
        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()  # e.g., 'expert_dataset'
        perturbation_level = parts[-1]  # e.g., '20'

        # load existing model
        model_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/iql_model_{perturbation_level}.pth"

        # ensure model exists
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"Model file not found: {model_save_path}. Cannot continue training.")
        else:
            print(f"Loading existing model from {model_save_path}")

        iql_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)

        # create environment
        iql_env, iql_action_dim = create_env(iql_logger, seed, env_id, True, "trial_1_continued")

        # initialize and load model
        iql_agent = IQLModel(iql_action_dim, device, seed, 0)
        iql_agent.load_state_dict(torch.load(model_save_path, map_location=device))
        iql_agent.to(device)

        # load existing stats
        stats_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/stats_{perturbation_level}.pkl"

        # ensure stats exist
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Training stats file not found: {stats_path}. Cannot continue training.")
        else:
            print(f"Loading existing stats from {stats_path}")
            with open(stats_path, 'rb') as f:
                existing_stats = pickle.load(f)

        # extract existing training progress
        prev_epochs = len(existing_stats['actor_losses'][0])
        total_epochs = prev_epochs + further_epochs

        criterion = nn.CrossEntropyLoss()

        trial_idx = 0  # since we're continuing only trial 1

        # get the existing q_values dictionary
        epoch_q_values = existing_stats.get('q_values', {
            trial_idx: {
                'avg_pred_q_values': {},
                'avg_target_q_values': {}
            }
        })

        # begin continued training loop
        for epoch in tqdm(range(prev_epochs, total_epochs), desc='Continued Training Epochs'):
            iql_agent.train()
            actor_loss, critic1_loss, critic2_loss, value_loss, avg_pred_q_values, avg_target_q_values = train_one_IQL_epoch(iql_agent, loaders['train'])

            # log training loss
            main_tag = f'{dataset_name}_Trial_1_Continued'
            iql_logger.log(f'{main_tag}/Actor Loss', actor_loss, epoch + 1)
            iql_logger.log(f'{main_tag}/Critic 1 Loss', critic1_loss, epoch + 1)
            iql_logger.log(f'{main_tag}/Critic 2 Loss', critic2_loss, epoch + 1)
            iql_logger.log(f'{main_tag}/Value Loss', value_loss, epoch + 1)

            # append new losses
            existing_stats['actor_losses'][0].append(actor_loss)
            existing_stats['critic1_losses'][0].append(critic1_loss)
            existing_stats['critic2_losses'][0].append(critic2_loss)
            existing_stats['value_losses'][0].append(value_loss)

            # add q_values for this epoch
            epoch_q_values[trial_idx]['avg_pred_q_values'][epoch] = avg_pred_q_values
            epoch_q_values[trial_idx]['avg_target_q_values'][epoch] = avg_target_q_values

            # tuning phase
            tuning_loss = evaluate_IQL_loss(iql_agent, criterion, loaders['tuning'], device)
            iql_logger.log(f'{main_tag}/Tune Loss', tuning_loss, epoch + 1)
            adjust_learning_rate(iql_agent, 0.98 if tuning_loss > 0.5 else 1.0)

            # Test phase
            test_loss = evaluate_IQL_loss(iql_agent, criterion, loaders['test'], device)
            iql_logger.log(f'{main_tag}/Test Loss', test_loss, epoch + 1)
            existing_stats['test_losses'][0].append(test_loss)

            # evaluation phase
            reward = evaluate_IQL_reward(iql_env, iql_agent, n_episodes=20, device=device, trial_idx=0, max_steps_per_episode=5000)
            iql_logger.log(f'{main_tag}/Reward', reward[0], epoch + 1)
            existing_stats['rewards'][0].append(reward)

            # free Memory After Each Epoch
            torch.cuda.empty_cache()
            gc.collect()
        
        # update q_values in the stats dict
        existing_stats['q_values'] = epoch_q_values

        print(f"Finished additional Training on {dataset_name} - Actor Loss: {actor_loss:.5f}")
        print(f"                                                              - Critic 1 Loss: {critic1_loss:.5f}")
        print(f"                                                              - Critic 2 Loss: {critic2_loss:.5f}")
        print(f"                                                              - Value Loss: {value_loss:.5f}")
        print(f"                                                              - Tuning Loss: {tuning_loss:.5f}")
        print(f"                                                              - Test Loss: {test_loss:.5f}")
        print(f"                                                              - Reward: {reward[0]:.2f}")

        # save updated model with continued name
        continued_model_path = model_save_path.replace('.pth', '_continued.pth')
        torch.save(iql_agent.state_dict(), continued_model_path)
        print(f"Updated DQN model saved to {continued_model_path}")

        # save updated stats with continued name
        continued_stats_path = stats_path.replace('.pkl', '_continued.pkl')
        with open(continued_stats_path, 'wb') as f:
            pickle.dump(existing_stats, f)
        print(f"Updated stats saved to {continued_stats_path}")

        iql_env.close()
    iql_logger.close()
