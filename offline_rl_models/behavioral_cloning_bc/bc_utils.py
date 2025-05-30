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
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import importlib

import auxiliary_methods.tb_logger
importlib.reload(auxiliary_methods.tb_logger)
from auxiliary_methods.tb_logger import TensorBoardLogger

import auxiliary_methods.utils
importlib.reload(auxiliary_methods.utils)
from auxiliary_methods.utils import create_env, save_return_stats, free_up_memory

import offline_rl_models.behavioral_cloning_bc.bc_model
importlib.reload(offline_rl_models.behavioral_cloning_bc.bc_model)
from offline_rl_models.behavioral_cloning_bc.bc_model import BCModel
    

def evaluate_BC_reward(env, model, n_episodes, device, max_steps_per_episode=1000, debug=False):
    """Evaluate the model by running it in the environment."""
    rewards = []

    for episode in range(n_episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                state = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                logits = model(state)
                action = torch.argmax(logits, dim=1).item()
                steps += 1

            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

        if debug:
            print(f"Finished Episode {episode+1}: Total Reward = {total_reward}, Steps Taken = {steps}")

        free_up_memory([state, logits, action, observation, reward, terminated, truncated])

    return rewards

def train_and_evaluate_BC(dataloaders, device, seeds, epochs, dataset, env_id, seed):
    logdir = 'agent_methods/behavioral_cloning_bc/bc_logs'
    stats_to_save = {}

    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training BC on {dataset_name}")
        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()
        perturbation_level = parts[-1]

        all_train_losses, all_val_losses, all_rewards = [], [], []
        all_train_steps, all_val_steps, all_reward_steps = [], [], []

        for seed_idx in range(seeds):
            print(f"-- Starting Seed {seed_idx + 1}/{seeds} --")

            bc_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)
            bc_env, bc_action_dim = create_env(bc_logger, seed, env_id, True, seed_number=f'{seed_idx+1}')
            bc_model = BCModel(bc_action_dim, seed, seed_idx).to(device)

            optimizer = optim.Adam(bc_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, min_lr=1e-5)

            main_tag = f'{dataset_name}_Seed_{seed_idx + 1}'

            train_losses, val_losses, rewards = [], [], []
            train_steps, val_steps, reward_steps = [], [], []
            train_step = val_step = reward_step = 0

            for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
                # ---- Train Phase ----
                bc_model.train()
                for states, actions, *_ in loaders['train']:
                    states, actions = states.to(device), actions.to(device)
                    optimizer.zero_grad()
                    outputs = bc_model(states)
                    loss = criterion(outputs, actions)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss)
                    train_steps.append(train_step)
                    bc_logger.log(f'{main_tag}/Train Loss', loss, train_step)
                    train_step += 1

                    free_up_memory([states, actions, outputs, loss])

                # ---- Validation Phase ----
                bc_model.eval()
                epoch_val_losses = []
                with torch.no_grad():
                    for states, actions, *_ in loaders['validation']:
                        states, actions = states.to(device), actions.to(device)
                        logits = bc_model(states)
                        loss = criterion(logits, actions)

                        val_losses.append(loss)
                        epoch_val_losses.append(loss)
                        val_steps.append(val_step)
                        bc_logger.log(f'{main_tag}/Validation Loss', loss, val_step)
                        val_step += 1

                        free_up_memory([states, actions, logits, loss])

                if epoch_val_losses:
                    scheduler.step(np.mean([loss.item() for loss in epoch_val_losses]))

                # ---- Reward Evaluation ----
                reward_list = evaluate_BC_reward(bc_env, bc_model, n_episodes=15, device=device, max_steps_per_episode=3000)
                for r in reward_list:
                    rewards.append(r)
                    reward_steps.append(reward_step)
                    bc_logger.log(f'{main_tag}/Reward', r, reward_step)
                    reward_step += 1

            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_rewards.append(rewards)
            all_train_steps.append(train_steps)
            all_val_steps.append(val_steps)
            all_reward_steps.append(reward_steps)

            print(f"Finished Training on {dataset_name}")
            print(f"    ➤ Avg Train Loss: {np.mean(train_losses):.5f}")
            print(f"    ➤ Avg Validation Loss: {np.mean(val_losses):.5f}")
            print(f"    ➤ Avg Reward: {np.mean(rewards):.2f}")

            # Save the model
            print("Saving the model...")
            model_save_path = f"{logdir}/{dataset_type}/{perturbation_level}/bc_model_{perturbation_level}_seed{seed_idx + 1}.pth"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(bc_model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            bc_env.close()
            time.sleep(1.5)

        stats_to_save = {
            'train_losses': all_train_losses,
            'val_losses': all_val_losses,
            'rewards': all_rewards,
            'train_steps': all_train_steps,
            'val_steps': all_val_steps,
            'reward_steps': all_reward_steps,
            'dataset_name': dataset_name,
            'num_seeds': seeds
        }

    stats_save_path = f"{logdir}/{dataset_type}/{perturbation_level}/stats_{perturbation_level}.pkl"
    os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)
    save_return_stats(stats_to_save, stats_save_path)
    print(f"Return Stats saved to {stats_save_path}")

    bc_logger.close()
