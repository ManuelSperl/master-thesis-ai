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
from auxiliary_methods.utils import create_env, save_return_stats, free_up_memory

import agent_methods.implicit_q_learning_iql.iql_model
importlib.reload(agent_methods.implicit_q_learning_iql.iql_model)
from agent_methods.implicit_q_learning_iql.iql_model import IQLModel

def evaluate_IQL_reward(env, agent, n_episodes, device, max_steps_per_episode=1000, debug=False):
    ''' Evaluate the IQL agent on the environment for n_episodes '''
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
    Trains an Implicit Q-Learning (IQL) agent on the specified dataset.
    """
    logdir = 'agent_methods/implicit_q_learning_iql/iql_logs'
    stats_to_save = {}

    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training IQL on {dataset_name}")

        # Extract metadata
        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()  # e.g., 'expert_dataset'
        perturbation_level = parts[-1]  # e.g., '20'

        # Init logs
        all_actor_losses, all_critic1_losses, all_critic2_losses, all_value_losses = [], [], [], []
        all_tune_losses, all_test_losses, all_rewards = [], [], []
        all_actor_steps, all_critic1_steps, all_critic2_steps, all_value_steps = [], [], [], []
        all_tune_steps, all_test_steps, all_reward_steps = [], [], []
        q_values_by_seed = {}

        # loop through seeds
        for seed_idx in range(seeds):
            print(f"-- Starting Seed {seed_idx + 1}/{seeds} --")

            # ------- Setup Phase -------
            iql_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)
            iql_env, iql_action_dim = create_env(iql_logger, seed, env_id, True, seed_number=f'{seed_idx+1}')
            iql_agent = IQLModel(iql_action_dim, device, seed, seed_idx)
            criterion = nn.CrossEntropyLoss()

            main_tag = f'{dataset_name}_Seed_{seed_idx + 1}'

            actor_losses, critic1_losses, critic2_losses, value_losses = [], [], [], []
            tune_losses, test_losses, rewards = [], [], []
            actor_steps, critic1_steps, critic2_steps, value_steps = [], [], [], []
            tune_steps, test_steps, reward_steps = [], [], []
            q_values_by_seed[seed_idx] = {'avg_pred_q_values': {}, 'avg_target_q_values': {}}
            actor_step = critic1_step = critic2_step = value_step = q_step = test_step = tune_step = reward_step = 0

            # start Offline Training -> loop through epochs
            for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
                # ------- Training Phase -------
                iql_agent.train()
                for states, actions, rews, next_states, dones in loaders['train']:
                    states = states.to(device)
                    actions = actions.to(device)
                    rews = rews.unsqueeze(1).to(device)
                    next_states = next_states.to(device)
                    dones = dones.unsqueeze(1).to(device)

                    out = iql_agent.learn((states, actions, rews, next_states, dones))
                    a_loss, c1_loss, c1_q, c1_tgt, c2_loss, c2_q, c2_tgt, v_loss = out

                    log_a = np.log10(a_loss + 1e-8)
                    actor_losses.append(log_a)
                    actor_steps.append(actor_step)
                    iql_logger.log(f'{main_tag}/Actor Loss', log_a, actor_step)
                    actor_step += 1

                    log_c1 = np.log10(c1_loss + 1e-8)
                    critic1_losses.append(log_c1)
                    critic1_steps.append(critic1_step)
                    iql_logger.log(f'{main_tag}/Critic 1 Loss', log_c1, critic1_step)
                    critic1_step += 1

                    log_c2 = np.log10(c2_loss + 1e-8)
                    critic2_losses.append(log_c2)
                    critic2_steps.append(critic2_step)
                    iql_logger.log(f'{main_tag}/Critic 2 Loss', log_c2, critic2_step)
                    critic2_step += 1

                    log_v = np.log10(v_loss + 1e-8)
                    value_losses.append(log_v)
                    value_steps.append(value_step)
                    iql_logger.log(f'{main_tag}/Value Loss', log_v, value_step)
                    value_step += 1

                    # Log Q-values with a shared step (e.g., critic1_step)
                    q_step = critic1_step
                    q_values_by_seed[seed_idx]['avg_pred_q_values'][q_step] = [(c1_q.mean().item() + c2_q.mean().item()) / 2]
                    q_values_by_seed[seed_idx]['avg_target_q_values'][q_step] = [(c1_tgt.mean().item() + c2_tgt.mean().item()) / 2]

                    free_up_memory([states, actions, rews, next_states, dones])

                # ---- Tune Phase ----
                iql_agent.actor.eval()
                epoch_tune_losses = []
                with torch.no_grad():
                    for states, actions, _, _, _ in loaders['tuning']:
                        states, actions = states.to(device), actions.to(device)
                        logits = iql_agent.actor(states)
                        loss = criterion(logits, actions)

                        log_loss = np.log10(loss.item() + 1e-8)
                        tune_losses.append(log_loss)
                        epoch_tune_losses.append(log_loss)
                        tune_steps.append(tune_step)
                        iql_logger.log(f'{main_tag}/Tune Loss', log_loss, tune_step)
                        tune_step += 1

                        free_up_memory([states, actions, logits, loss])

                if epoch_tune_losses:
                    for opt in [iql_agent.actor_optimizer, iql_agent.critic1_optimizer, iql_agent.critic2_optimizer, iql_agent.value_optimizer]:
                        for pg in opt.param_groups:
                            pg['lr'] *= 0.98 if np.mean(epoch_tune_losses) > 0.5 else 1.0

                # ---- Test Phase ----
                with torch.no_grad():
                    for states, actions, _, _, _ in loaders['test']:
                        states, actions = states.to(device), actions.to(device)
                        logits = iql_agent.actor(states)
                        loss = criterion(logits, actions)

                        log_loss = np.log10(loss.item() + 1e-8)
                        test_losses.append(log_loss)
                        test_steps.append(test_step)
                        iql_logger.log(f'{main_tag}/Test Loss', log_loss, test_step)
                        test_step += 1

                        free_up_memory([states, actions, logits, loss])

                # ---- Reward Phase ----
                reward_list = evaluate_IQL_reward(iql_env, iql_agent, n_episodes=15, device=device, max_steps_per_episode=3000)
                for r in reward_list:
                    rewards.append(r)
                    reward_steps.append(reward_step)
                    iql_logger.log(f'{main_tag}/Reward', r, reward_step)
                    reward_step += 1

            
            # Store all seed data
            all_actor_losses.append(actor_losses)
            all_critic1_losses.append(critic1_losses)
            all_critic2_losses.append(critic2_losses)
            all_value_losses.append(value_losses)
            all_tune_losses.append(tune_losses)
            all_test_losses.append(test_losses)
            all_rewards.append(rewards)
            all_actor_steps.append(actor_steps)
            all_critic1_steps.append(critic1_steps)
            all_critic2_steps.append(critic2_steps)
            all_value_steps.append(value_steps)
            all_tune_steps.append(tune_steps)
            all_test_steps.append(test_steps)
            all_reward_steps.append(reward_steps)

            print(f"Finished Training on {dataset_name}")
            print(f"    ➤ Avg Actor Loss: {np.mean(actor_losses):.5f}")
            print(f"    ➤ Avg Critic1 Loss: {np.mean(critic1_losses):.5f}")
            print(f"    ➤ Avg Critic2 Loss: {np.mean(critic2_losses):.5f}")
            print(f"    ➤ Avg Value Loss: {np.mean(value_losses):.5f}")
            print(f"    ➤ Avg Test Loss: {np.mean(test_losses):.5f}")
            print(f"    ➤ Avg Reward: {np.mean(rewards):.2f}")

            if seed_idx == 0:
                model_path = f"{logdir}/{dataset_type}/{perturbation_level}/iql_model_{perturbation_level}.pth"
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
            'tune_losses': all_tune_losses,
            'test_losses': all_test_losses,
            'rewards': all_rewards,
            'actor_steps': all_actor_steps,
            'critic1_steps': all_critic1_steps,
            'critic2_steps': all_critic2_steps,
            'value_steps': all_value_steps,
            'tune_steps': all_tune_steps,
            'test_steps': all_test_steps,
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
