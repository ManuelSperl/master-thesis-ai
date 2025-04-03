# bc_utils.py contains utility functions for training and evaluating the Behavioral Cloning model.

# ----- PyTorch imports -----
import torch
import torch.nn as nn
import torch.optim as optim

# ----- Python imports -----
import gc
import os
import pickle
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import importlib

import auxiliary_methods.tb_logger
importlib.reload(auxiliary_methods.tb_logger)
from auxiliary_methods.tb_logger import TensorBoardLogger

import auxiliary_methods.utils
importlib.reload(auxiliary_methods.utils)
from auxiliary_methods.utils import create_env, save_return_stats

import behavioral_cloning_bc.bc_model
importlib.reload(behavioral_cloning_bc.bc_model)
from behavioral_cloning_bc.bc_model import BCModel

def train_one_BC_epoch(model, loader, criterion, optimizer, device):
    """Trains BC for one epoch using a supervised learning approach."""
    model.train()
    running_loss = 0.0

    if len(loader) == 0:
        return 0.0  # ensure we don't divide by zero

    for states, actions, _, _, _ in loader:
        states, actions = states.to(device), actions.to(device)

        optimizer.zero_grad()

        # compute logits (predicted actions)
        outputs = model(states)

        # compute loss
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # free memory after each batch
        del states, actions, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

    return running_loss / len(loader)

def evaluate_BC_loss(model, criterion, data_loader, device):
    """
    Evaluate the BC model.

    :param model: the BC model
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
        for (states, actions, _, _, _) in data_loader:
            states, actions = states.to(device), actions.to(device).long()

            # forward pass
            logits = model(states)
            loss = criterion(logits, actions)

            total_loss += loss.item() * states.size(0)
            total_samples += states.size(0)

            # free memory after each episode
            del states, actions, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

    return total_loss / total_samples if total_samples > 0 else 0.0

def evaluate_BC_reward(env, model, n_episodes, device, trial_idx, max_steps_per_episode=1000, debug=False):
    """Evaluate the model by running it in the environment."""
    rewards = []

    # make sure environment gets a unique seed per trial
    env.seed(1234 + trial_idx)  
    env.action_space.seed(1234 + trial_idx)

    for episode in range(n_episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0  # step counter

        # set frames_per_sec if the video recorder exists
        video_recorder = env.get_wrapper_attr('video_recorder')
        if video_recorder is not None:
            video_recorder.frames_per_sec = 30

        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                # process observation
                state = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                logits = model(state)
                action = torch.argmax(logits, dim=1).item()
                steps += 1  # increase step count

            # take the action in the environment
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if debug:
                if steps % 500 == 0:
                    print(f"Episode {episode+1}: {steps} steps taken, Total Reward: {total_reward}")

            # set frames_per_sec after the first step (if needed)
            if video_recorder is not None:
                video_recorder.frames_per_sec = 30

        rewards.append(total_reward)

        if debug:
            print(f"Finished Episode {episode+1}: Total Reward = {total_reward}, Steps Taken = {steps}")

        # free memory after each episode
        del state, action, observation, reward, terminated, truncated
        torch.cuda.empty_cache()
        gc.collect()

    return [np.mean(rewards)]

def train_and_evaluate_BC(dataloaders, device, trials, epochs, dataset, env_id, seed):
    """
    Train and evaluate the BC model, logging loss and reward for each epoch.

    :param dataloaders: dictionary containing the dataloaders for each dataset
    :param loggerpath: the path for the logger
    :param device: the device
    :param trials: the number of trials
    :param epochs: the number of epochs
    :param dataset: the dataset to train on
    """
    logdir = 'behavioral_cloning_bc/bc_logs'
    stats_to_save = {}

    # train and evaluate the BC model on all datasets
    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training on {dataset_name}")

        # split the key to extract dataset type and perturbation level
        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()  # e.g. 'expert_dataset'
        perturbation_level = parts[-1]      # e.g. '20'

        # lists to store losses and rewards for plotting
        all_train_losses, all_test_losses, all_rewards = [], [], []

        # loop through trials
        for trial in range(trials):
            print(f"-- Starting Trial {trial + 1}/{trials} --")

            # ------- Setup Phase -------
            # logger setup
            bc_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)

            # create the Enviornment
            bc_env, bc_action_dim = create_env(bc_logger, seed, env_id, True, trial_number=f'{trial+1}')

            # initialize the BC model, optimizer, loss function, and scheduler
            bc_model = BCModel(bc_action_dim, seed, trial).to(device)
            optimizer = optim.Adam(bc_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, min_lr=1e-5)

            # lists to store losses and rewards for this trial
            train_losses, test_losses, rewards = [], [], []

            # print initial hyperparameters
            initial_lr = optimizer.param_groups[0]['lr']
            lr_updates = []  # store any learning rate updates
            lr_updates.append(f"Initial learning rate: {initial_lr:.6f}")

            # start Offline Training -> loop through epochs
            for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
                # ------- Training Phase -------
                bc_model.train()
                train_loss = train_one_BC_epoch(bc_model, loaders['train'], criterion, optimizer, device)
                train_losses.append(train_loss)

                # log training loss
                main_tag = f'{dataset_name}_Trial_{trial + 1}'
                bc_logger.log(f'{main_tag}/Train Loss', train_loss, epoch + 1)

                # ------- Hyperparameter Tuning Phase -------
                tune_loss = evaluate_BC_loss(bc_model, criterion, loaders['tuning'], device)
                bc_logger.log(f'{main_tag}/Tune Loss', tune_loss, epoch + 1)

                # adjust learning rate based on tuning loss
                scheduler.step(tune_loss)

                # print adjusted hyperparameters
                adjusted_lr = optimizer.param_groups[0]['lr']
                if adjusted_lr != initial_lr:
                    lr_updates.append(f"Adjusted learning rate in epoch {epoch + 1}: {adjusted_lr:.6f}")
                    initial_lr = adjusted_lr  # update for next comparison

                # ------- Test Phase -------
                test_loss = evaluate_BC_loss(bc_model, criterion, loaders['test'], device)
                test_losses.append(test_loss)
                bc_logger.log(f'{main_tag}/Test Loss', test_loss, epoch + 1)

                # evaluation and reward collection at the end of each epoch
                reward = evaluate_BC_reward(bc_env, bc_model, n_episodes=20, device=device, trial_idx=trial, max_steps_per_episode=5000)
                rewards.extend(reward)
                bc_logger.log(f'{main_tag}/Reward', reward[0], epoch + 1) # log reward

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
                model_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/bc_model_{perturbation_level}.pth"
                torch.save(bc_model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

            # close the enviornment after each trial
            bc_env.close()
            time.sleep(1.5)  # give filesystem time to flush the metadata file
        
        stats_to_save = {
            'train_losses': all_train_losses,
            'test_losses': all_test_losses,
            'rewards': all_rewards,
            'dataset_name': dataset_name,
            'trials': trials
        }

    # save the return stats
    stats_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/stats_{perturbation_level}.pkl"
    save_return_stats(stats_to_save, stats_save_path)
    print(f"Return Stats saved to {stats_save_path}")

    # close the logger
    bc_logger.close()

def continue_training_BC(dataloaders, further_epochs, dataset, env_id, seed, device):
    """
    Continues training an existing BC model from a previous trial.

    :param dataloaders: Dataset loaders.
    :param further_epochs: Number of additional epochs.
    :param dataset: Dataset name.
    :param env_id: Environment ID.
    :param seed: Seed for reproducibility.
    :param device: Training device.
    """
    logdir = 'behavioral_cloning_bc/bc_logs'

    # loop through datasets
    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Continuing BC Training on {dataset_name}")

        # split dataset name to extract type and perturbation level
        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()  # e.g., 'expert_dataset'
        perturbation_level = parts[-1]  # e.g., '20'

        # load model
        model_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/bc_model_{perturbation_level}.pth"
        
        # ensure model exists
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"Model file not found: {model_save_path}. Cannot continue training.")
        else:
            print(f"Loading existing model from {model_save_path}")

        bc_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)

        # create environment
        bc_env, bc_action_dim = create_env(bc_logger, seed, env_id, True, "trial_1_continued")

        # initialize and load model
        bc_model = BCModel(bc_action_dim, seed, 0).to(device)
        bc_model.load_state_dict(torch.load(model_save_path, map_location=device))

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
        prev_epochs = len(existing_stats['train_losses'][0])
        total_epochs = prev_epochs + further_epochs

        optimizer = optim.Adam(bc_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9)

        # training Loop
        for epoch in tqdm(range(prev_epochs, total_epochs), desc='Continued Training Epochs'):
            train_loss = train_one_BC_epoch(bc_model, loaders['train'], criterion, optimizer, device)

            # log training loss
            main_tag = f'{dataset_name}_Trial_1_Continued'
            bc_logger.log(f'{main_tag}/Train Loss', train_loss, epoch + 1)
            existing_stats['train_losses'][0].append(train_loss)

            # Tuning phase
            tuning_loss = evaluate_BC_loss(bc_model, criterion, loaders['tuning'], device)
            bc_logger.log(f'{main_tag}/Tune Loss', tuning_loss, epoch + 1)
            scheduler.step(tuning_loss)

            # Test phase
            test_loss = evaluate_BC_loss(bc_model, criterion, loaders['test'], device)
            bc_logger.log(f'{main_tag}/Test Loss', test_loss, epoch + 1)
            existing_stats['test_losses'][0].append(test_loss)

            # evaluation and reward collection at the end of each epoch
            reward = evaluate_BC_reward(bc_env, bc_model, n_episodes=20, device=device, trial_idx=0, max_steps_per_episode=5000)
            bc_logger.log(f'{main_tag}/Reward', reward[0], epoch + 1)
            existing_stats['rewards'][0].append(reward)

            # free Memory After Each Epoch
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"Finished additional Training on {dataset_name} - Training Loss: {train_loss:.5f}")
        print(f"                                                              - Tuning Loss: {tuning_loss:.5f}")
        print(f"                                                              - Test Loss: {test_loss:.5f}")
        print(f"                                                              - Reward: {reward[0]:.2f}")

        # save updated model with continued name
        continued_model_path = model_save_path.replace('.pth', '_continued.pth')
        torch.save(bc_model.state_dict(), continued_model_path)
        print(f"Updated DQN model saved to {continued_model_path}")

        # save updated stats with continued name
        continued_stats_path = stats_path.replace('.pkl', '_continued.pkl')
        with open(continued_stats_path, 'wb') as f:
            pickle.dump(existing_stats, f)
        print(f"Updated stats saved to {continued_stats_path}")

        bc_env.close()
    bc_logger.close()
