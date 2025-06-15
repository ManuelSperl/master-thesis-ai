# bve_utils.py 

import torch
import numpy as np
import os
import time
from tqdm import tqdm
import torch.nn.functional as F

import importlib

import auxiliary_methods.tb_logger
importlib.reload(auxiliary_methods.tb_logger)
from auxiliary_methods.tb_logger import TensorBoardLogger

import auxiliary_methods.utils
importlib.reload(auxiliary_methods.utils)
from auxiliary_methods.utils import create_env, save_return_stats, free_up_memory

import offline_rl_models.behavior_value_estimation_bve.bve_model
importlib.reload(offline_rl_models.behavior_value_estimation_bve.bve_model)
from offline_rl_models.behavior_value_estimation_bve.bve_model import BVEModel

def evaluate_BVE_reward(env, agent, n_episodes, device, max_steps_per_episode=1000, debug=False):
    rewards = []    
    agent.eval()

    for episode in range(n_episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                state = torch.tensor(observation, dtype=torch.float32)
                state = state.permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                action = agent.get_action(state).squeeze().item()

            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        if debug:
            print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

        free_up_memory([state, observation, reward, terminated, truncated])

    return rewards

def train_and_evaluate_BVE(dataloaders, epochs, seeds, dataset, env_id, seed, device):
    logdir = 'offline_rl_models/behavior_value_estimation_bve/bve_logs'
    stats_to_save = {}

    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training BVE on {dataset_name}")

        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()
        perturbation_level = parts[-1]

        all_train_losses, all_val_losses, all_rewards = [], [], []
        all_train_steps, all_val_steps, all_reward_steps = [], [], []
        q_values_by_seed = {}

        for seed_idx in range(seeds):
            print(f"-- Starting Seed {seed_idx + 1}/{seeds} --")

            bve_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)
            bve_env, bve_action_dim = create_env(bve_logger, seed, env_id, True, seed_number=f"{seed_idx + 1}")
            bve_agent = BVEModel(bve_action_dim, device, seed, seed_idx)

            main_tag = f'{dataset_name}_Seed_{seed_idx + 1}'
            train_losses, val_losses, rewards = [], [], []
            train_steps, val_steps, reward_steps = [], [], []
            train_step = val_step = reward_step = 0
            q_values_by_seed[seed_idx] = {'avg_pred_q_values': {}, 'avg_target_q_values': {}}

            for epoch in tqdm(range(epochs), desc='Epochs'):
                # ---- Training Phase ----
                bve_agent.train()
                for states, actions, rews, next_states, next_actions, dones, *_ in loaders['train']:
                    states, actions = states.to(device), actions.to(device)
                    rews = rews.unsqueeze(1).to(device)
                    next_states = next_states.to(device)
                    next_actions = next_actions.to(device)
                    dones = dones.unsqueeze(1).to(device)

                    loss, pred_q, target_q = bve_agent.learn((states, actions, rews, next_states, next_actions, dones))

                    train_losses.append(loss)
                    train_steps.append(train_step)
                    bve_logger.log(f'{main_tag}/Train Loss', loss, train_step)
                    train_step += 1

                    q_values_by_seed[seed_idx]['avg_pred_q_values'][train_step] = [pred_q.mean().item()]
                    q_values_by_seed[seed_idx]['avg_target_q_values'][train_step] = [target_q.mean().item()]

                    free_up_memory([states, actions, rews, next_states, next_actions, dones])
                
                # ---- Validation Phase ----
                bve_agent.eval()
                epoch_val_losses = []
                with torch.no_grad():
                    for states, actions, rews, next_states, next_actions, dones, *_ in loaders['validation']:
                        states, actions = states.to(device), actions.to(device)
                        rews = rews.unsqueeze(1).to(device)
                        next_states = next_states.to(device)
                        next_actions = next_actions.to(device)
                        dones = dones.unsqueeze(1).to(device)

                        target_q = bve_agent.target_q_net(next_states, next_actions)
                        target = rews + bve_agent.gamma * (1 - dones) * target_q
                        current_q = bve_agent.q_net(states, actions)
                        loss = F.smooth_l1_loss(current_q, target)

                        val_losses.append(loss.item())
                        epoch_val_losses.append(loss.item())
                        val_steps.append(val_step)
                        bve_logger.log(f'{main_tag}/Val Loss', loss.item(), val_step)
                        val_step += 1

                        free_up_memory([states, actions, rews, next_states, next_actions, dones])
                
                # -- Learning rate decay based on validation loss --
                if epoch_val_losses:
                    for opt in [bve_agent.optimizer]:
                        for pg in opt.param_groups:
                            pg['lr'] *= 0.98 if np.mean(epoch_val_losses) > 0.5 else 1.0
                
                # ---- Reward Evaluation ----
                reward_list = evaluate_BVE_reward(bve_env, bve_agent, n_episodes=15, device=device, max_steps_per_episode=3000)
                for r in reward_list:
                    rewards.append(r)
                    reward_steps.append(reward_step)
                    bve_logger.log(f'{main_tag}/Reward', r, reward_step)
                    reward_step += 1

            # Save data
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_rewards.append(rewards)
            all_train_steps.append(train_steps)
            all_val_steps.append(val_steps)
            all_reward_steps.append(reward_steps)

            print(f"Finished Training on {dataset_name}")
            print(f"    ➤ Avg Train Loss: {np.mean(train_losses):.5f}")
            print(f"    ➤ Avg Val Loss: {np.mean(val_losses):.5f}")
            print(f"    ➤ Avg Reward: {np.mean(rewards):.2f}")

            # Save the model
            print("Saving the model...")
            model_path = f"{logdir}/{dataset_type}/{perturbation_level}/bve_model_{perturbation_level}_seed{seed_idx + 1}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(bve_agent.state_dict(), model_path)
            print(f"Saved model to {model_path}")

            bve_env.close()
            time.sleep(1.5)

        stats_to_save = {
            'train_losses': all_train_losses,
            'val_losses': all_val_losses,
            'rewards': all_rewards,
            'train_steps': all_train_steps,
            'val_steps': all_val_steps,
            'reward_steps': all_reward_steps,
            'q_values': q_values_by_seed,
            'dataset_name': dataset_name,
            'num_seeds': seeds
        }

    stats_path = f"{logdir}/{dataset_type}/{perturbation_level}/stats_{perturbation_level}.pkl"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    save_return_stats(stats_to_save, stats_path)
    print(f"Saved stats to {stats_path}")

    bve_logger.close()