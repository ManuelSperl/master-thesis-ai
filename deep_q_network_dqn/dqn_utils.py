# dqn_utils.py

import torch
import torch.nn as nn
import torch.optim as optim

import gc
import os
import pickle
import numpy as np
from tqdm import tqdm

import importlib

import auxiliary_methods.tb_logger
importlib.reload(auxiliary_methods.tb_logger)
from auxiliary_methods.tb_logger import TensorBoardLogger

import auxiliary_methods.utils
importlib.reload(auxiliary_methods.utils)
from auxiliary_methods.utils import create_env, save_return_stats

import datasets.dataset_utils
importlib.reload(datasets.dataset_utils)

from datasets.dataset_utils import set_all_seeds

import deep_q_network_dqn.dqn_model
importlib.reload(deep_q_network_dqn.dqn_model)
from deep_q_network_dqn.dqn_model import DQNModel

import deep_q_network_dqn.dqn_replay_buffer
importlib.reload(deep_q_network_dqn.dqn_replay_buffer)
from deep_q_network_dqn.dqn_replay_buffer import ReplayBuffer

def train_one_DQN_epoch(model, target_net, replay_buffer, criterion, optimizer, device, updates_per_epoch):
    """Trains DQN for one epoch using experience replay."""
    model.train()
    running_loss = 0.0

    if len(replay_buffer) < replay_buffer.batch_size:
        return 0.0  # not enough samples yet

    for _ in range(updates_per_epoch):
        states, actions, rewards, next_states, dones = replay_buffer.sample()

        optimizer.zero_grad()

        # compute Q-values for current states
        q_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # compute target Q-values using the target network
        with torch.no_grad():
            best_next_actions = model(next_states).argmax(dim=1, keepdim=True)
            next_q_values = target_net(next_states).gather(1, best_next_actions).squeeze(-1)
            target_q_values = rewards + (0.99 * next_q_values * (1 - dones))

        loss = criterion(q_values, target_q_values)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # free memory after each episode
        del states, actions, rewards, next_states, dones, q_values, next_q_values, target_q_values
        torch.cuda.empty_cache()
        gc.collect()

    return running_loss / updates_per_epoch

def evaluate_DQN_loss(model, criterion, data_loader, device):
    """
    Evaluate the DQN model.

    :param model: the DQN model
    :param criterion: the loss function
    :param data_loader: the DataLoader object
    :param device: the device

    :return: the average loss for the evaluation
    """
    model.eval() # set model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    
    # compute evaluation Loss from dataset
    with torch.no_grad():
        for states, actions, rewards, next_states, dones in data_loader:
            states, actions = states.to(device), actions.to(device).long()
            rewards, next_states, dones = rewards.to(device), next_states.to(device), dones.to(device)

            # compute Q-values for current states
            q_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # compute target Q-values using the main network
            best_next_actions = model(next_states).argmax(dim=1, keepdim=True)
            next_q_values = model(next_states).gather(1, best_next_actions).squeeze(-1)
            target_q_values = rewards + (0.99 * next_q_values * (1 - dones))

            # compute loss
            loss = criterion(q_values, target_q_values)
            total_loss += loss.item() * states.size(0)
            total_samples += states.size(0)

            # free memory after each episode
            del states, actions, rewards, next_states, dones, q_values, next_q_values, target_q_values
            torch.cuda.empty_cache()
            gc.collect()

    return total_loss / total_samples if total_samples > 0 else 0.0

def evaluate_DQN_reward(env, model, n_episodes, device, trial_idx, max_steps_per_episode=1000, debug=False):
    """Evaluate the model by running it in the environment."""
    rewards = []

    # make sure environment gets a unique seed per trial
    env.seed(1234 + trial_idx)  
    env.action_space.seed(1234 + trial_idx)

    action_counts = np.zeros(env.action_space.n)

    # set model to evaluation mode
    model.eval()

    for episode in range(n_episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0  # step counter

        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                state = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

                q_values = model(state)
                
                if eval == True and episode == 0 and steps == 0:
                    print("Q-values for first frame:", q_values.cpu().numpy())

                action = model.get_action(state, eval=True)
                action_counts[action] += 1

            # take the action in the environment
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1  # increase step count

        rewards.append(total_reward)
        
        if debug:
            print(f"Finished Episode {episode+1}: Total Reward = {total_reward}, Steps Taken = {steps}")
            print(f"Action Distribution: {action_counts}")

        # free memory after each episode
        del state, action, observation, reward, terminated, truncated
        torch.cuda.empty_cache()
        gc.collect()

    return [np.mean(rewards)]

def train_and_evaluate_DQN(dataloaders, device, trials, epochs, dataset, env_id, seed):
    """Train and evaluate the DQN model, logging loss and reward for each epoch."""

    logdir = 'deep_q_network_dqn/dqn_logs'
    stats_to_save = {}

    # train and evaluate the DQN model on all datasets
    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training on {dataset_name}")

        # split the key to extract dataset type and perturbation level
        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()  # e.g. 'expert_dataset'
        perturbation_level = parts[-1]      # e.g. '20'

        # lists to store losses and rewards for plotting
        all_train_losses, all_test_losses, all_rewards = [], [], []

        # initialize replay buffer with a large capacity
        replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=32, device=device)
        print("Filling replay buffer from dataset...")

        if len(replay_buffer) == 0: # fill buffer only once
            for states, actions, rewards, next_states, dones in loaders['train']:
                for i in range(states.shape[0]):
                    replay_buffer.add(
                        states[i],
                        actions[i].item(),
                        rewards[i].item(),
                        next_states[i],
                        dones[i].item()
                    )
            print(f"Replay buffer filled with {len(replay_buffer)} samples.")

        # loop through trials
        for trial in range(trials):
            print(f"-- Starting Trial {trial + 1}/{trials} --")

            set_all_seeds(seed + trial) # set all seeds for reproducibility

            # ------- Setup Phase -------
            # logger setup
            dqn_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)

            # create the Enviornment
            dqn_env, dqn_action_dim = create_env(dqn_logger, seed, env_id, True, trial_number=f'{trial+1}')

            # initialize DQN and Target Network
            dqn_model = DQNModel(dqn_action_dim, seed, trial).to(device)
            target_net = DQNModel(dqn_action_dim, seed, trial).to(device)

            # copy initial weights from DQN to target_net
            target_net.load_state_dict(dqn_model.state_dict())
            target_net.eval()  # target network is frozen

            optimizer = optim.AdamW(dqn_model.parameters(), lr=5e-4, weight_decay=1e-4)
            criterion = nn.SmoothL1Loss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, min_lr=1e-5)
            tau = 0.02  # soft update parameter

            # lists to store losses and rewards for each trial
            train_losses, test_losses, rewards = [], [], []

            # print initial hyperparameters
            initial_lr = optimizer.param_groups[0]['lr']
            lr_updates = []  # store any learning rate updates
            lr_updates.append(f"Initial learning rate: {initial_lr:.6f}")

            # start Offline Training -> loop through epochs
            for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
                # ------- Training Phase -------
                dqn_model.train()
                updates_per_epoch = min(len(loaders['train']), 500)  # limit to 500 updates
                train_loss = train_one_DQN_epoch(dqn_model, target_net, replay_buffer, criterion, optimizer, device, updates_per_epoch)

                # soft update of target network parameters
                for target_param, local_param in zip(target_net.parameters(), dqn_model.parameters()):
                    target_param.data.copy_((1 - tau) * target_param.data + tau * local_param.data)

                # log training loss
                train_losses.append(train_loss)
                main_tag = f'{dataset_name}_Trial_{trial + 1}'
                dqn_logger.log(f'{main_tag}/Train Loss', train_loss, epoch + 1)

                # ------- Hyperparameter Tuning Phase -------
                tune_loss = evaluate_DQN_loss(dqn_model, criterion, loaders['tuning'], device)
                dqn_logger.log(f'{main_tag}/Tune Loss', tune_loss, epoch + 1)

                # adjust learning rate based on tuning loss
                scheduler.step(tune_loss)

                # print adjusted hyperparameters
                adjusted_lr = optimizer.param_groups[0]['lr']
                if adjusted_lr != initial_lr:
                    lr_updates.append(f"Adjusted learning rate in epoch {epoch + 1}: {adjusted_lr:.6f}")
                    initial_lr = adjusted_lr  # update for next comparison

                # ------- Test Phase -------
                test_loss = evaluate_DQN_loss(dqn_model, criterion, loaders['test'], device)
                test_losses.append(test_loss)
                dqn_logger.log(f'{main_tag}/Test Loss', test_loss, epoch + 1)

                # evaluation and reward collection at the end of each epoch
                reward = evaluate_DQN_reward(dqn_env, dqn_model, n_episodes=20, device=device, trial_idx=trial, max_steps_per_episode=5000)
                rewards.extend(reward)
                dqn_logger.log(f'{main_tag}/Epoch Reward', reward[0], epoch + 1) # log reward

                # free Memory After Each Epoch
                torch.cuda.empty_cache()
                gc.collect()
            
            # print all learning rate updates after epochs are completed
            for msg in lr_updates:
                print(msg)

            # collect losses and rewards for all trials
            all_train_losses.append(train_losses)
            all_test_losses.append(test_losses)
            all_rewards.append(rewards)

            # print final losses and rewards for this trial
            print(f"Finished Training on {dataset_name} - Training Loss: {train_loss:.5f}")
            print(f"                                                   - Tuning Loss: {tune_loss:.5f}")
            print(f"                                                   - Test Loss: {test_loss:.5f}")
            print(f"                                                   - Reward: {reward[0]:.2f}")

            # save model after the first trial
            if trial == 0:
                model_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/dqn_model_{perturbation_level}.pth"
                torch.save(dqn_model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

            # close the enviornment
            dqn_env.close()

        stats_to_save = {
            'train_losses': all_train_losses,
            'test_losses': all_test_losses,
            'rewards': all_rewards,
            'dataset_name': dataset_name,
            'trials': trials,
        }
    
    # save the return stats
    stats_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/stats_{perturbation_level}.pkl"
    save_return_stats(stats_to_save, stats_save_path)
    print(f"Return Stats saved to {stats_save_path}")

    # close the logger
    dqn_logger.close()

def continue_training_DQN(dataloaders, further_epochs, dataset, env_id, seed, device):
    """
    Continues training an existing DQN model from a previous trial.

    :param dataloaders: Dataset loaders.
    :param further_epochs: Number of additional epochs.
    :param dataset: Dataset name.
    :param env_id: Environment ID.
    :param seed: Seed for reproducibility.
    :param device: Training device.
    """
    logdir = 'deep_q_network_dqn/dqn_logs'

    # loop through datasets
    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Continuing DQN Training on {dataset_name}")

        # split dataset name to extract type and perturbation level
        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()  # e.g., 'expert_dataset'
        perturbation_level = parts[-1]  # e.g., '20'

        # load model
        model_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/dqn_model_{perturbation_level}.pth"

        # ensure model exists
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"Model file not found: {model_save_path}. Cannot continue training.")
        else:
            print(f"Loading existing model from {model_save_path}")

        dqn_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)

        # create environment
        dqn_env, dqn_action_dim = create_env(dqn_logger, seed, env_id, True, "trial_1_continued")

        # Initialize and load model
        dqn_model = DQNModel(dqn_action_dim, seed, trial_idx=0).to(device)
        dqn_model.load_state_dict(torch.load(model_save_path, map_location=device))

        # initialize target network and copy weights
        target_net = DQNModel(dqn_action_dim, seed, trial_idx=0).to(device)
        target_net.load_state_dict(dqn_model.state_dict())
        target_net.eval()

        # Load existing stats
        stats_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/stats_{perturbation_level}.pkl"

        # ensure stats exist
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Training stats file not found: {stats_path}. Cannot continue training.")
        else:
            print(f"Loading existing stats from {stats_path}")
            with open(stats_path, 'rb') as f:
                existing_stats = pickle.load(f)

        # extract existing training progress
        prev_epochs = len(existing_stats['train_losses'][0])
        total_epochs = prev_epochs + further_epochs

        # optimizer, loss function, scheduler
        optimizer = optim.AdamW(dqn_model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, min_lr=1e-5)
        tau = 0.02

        # replay buffer (refilled from dataset)
        replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=32, device=device)
        print("Filling replay buffer...")
        for states, actions, rewards, next_states, dones in loaders['train']:
            for i in range(states.shape[0]):
                replay_buffer.add(
                    states[i],
                    actions[i].item(),
                    rewards[i].item(),
                    next_states[i],
                    dones[i].item()
                )
        print(f"Replay buffer filled with {len(replay_buffer)} samples.")

        # begin continued training loop
        for epoch in tqdm(range(prev_epochs, total_epochs), desc='Continued DQN Epochs'):
            dqn_model.train()
            updates_per_epoch = min(len(loaders['train']), 500)
            train_loss = train_one_DQN_epoch(dqn_model, target_net, replay_buffer, criterion, optimizer, device, updates_per_epoch)

            # soft update target network
            for target_param, local_param in zip(target_net.parameters(), dqn_model.parameters()):
                target_param.data.copy_((1 - tau) * target_param.data + tau * local_param.data)

            # logging training loss
            main_tag = f'{dataset_name}_Trial_1_Continued'
            dqn_logger.log(f'{main_tag}/Train Loss', train_loss, epoch + 1)
            existing_stats['train_losses'][0].append(train_loss)

            # Tuning phase
            tuning_loss = evaluate_DQN_loss(dqn_model, criterion, loaders['tuning'], device)
            dqn_logger.log(f'{main_tag}/Tune Loss', tuning_loss, epoch + 1)
            scheduler.step(tuning_loss)

            # Test phase
            test_loss = evaluate_DQN_loss(dqn_model, criterion, loaders['test'], device)
            dqn_logger.log(f'{main_tag}/Test Loss', test_loss, epoch + 1)
            existing_stats['test_losses'][0].append(test_loss)

            # evaluation and reward collection at the end of each epoch
            reward = evaluate_DQN_reward(dqn_env, dqn_model, n_episodes=20, device=device, trial_idx=0, max_steps_per_episode=5000)
            dqn_logger.log(f'{main_tag}/Reward', reward[0], epoch + 1)
            existing_stats['rewards'][0].extend(reward)

            # free Memory After Each Epoch
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"Finished additional Training on {dataset_name} - Training Loss: {train_loss:.5f}")
        print(f"                                                              - Tuning Loss: {tuning_loss:.5f}")
        print(f"                                                              - Test Loss: {test_loss:.5f}")
        print(f"                                                              - Reward: {reward[0]:.2f}")

        # save updated model with continued name
        continued_model_path = model_save_path.replace('.pth', '_continued.pth')
        torch.save(dqn_model.state_dict(), continued_model_path)
        print(f"Updated DQN model saved to {continued_model_path}")

        # save updated stats with continued name
        continued_stats_path = stats_path.replace('.pkl', '_continued.pkl')
        with open(continued_stats_path, 'wb') as f:
            pickle.dump(existing_stats, f)
        print(f"Updated stats saved to {continued_stats_path}")

        dqn_env.close()
    dqn_logger.close()