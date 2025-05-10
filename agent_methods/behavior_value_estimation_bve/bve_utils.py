# iql_utils.py 

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

import agent_methods.behavior_value_estimation_bve.bve_model
importlib.reload(agent_methods.behavior_value_estimation_bve.bve_model)
from agent_methods.behavior_value_estimation_bve.bve_model import BVEModel

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
    logdir = 'agent_methods/behavior_value_estimation/bve_logs'
    stats_to_save = {}

    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training BVE on {dataset_name}")

        parts = dataset_name.split('_')
        dataset_type = '_'.join(parts[:2]).lower()
        perturbation_level = parts[-1]

        all_train_losses, all_test_losses, all_tune_losses, all_rewards = [], [], [], []
        all_train_steps, all_test_steps, all_tune_steps, all_reward_steps = [], [], [], []
        q_values_by_seed = {}

        for seed_idx in range(seeds):
            print(f"-- Starting Seed {seed_idx + 1}/{seeds} --")

            bve_logger = TensorBoardLogger(logdir, dataset_type, perturbation_level)
            bve_env, bve_action_dim = create_env(bve_logger, seed, env_id, True, seed_number=f"{seed_idx + 1}")
            bve_agent = BVEModel(bve_action_dim, device, seed, seed_idx)

            main_tag = f'{dataset_name}_Seed_{seed_idx + 1}'
            train_losses, test_losses, tune_losses, rewards = [], [], [], []
            train_steps, test_steps, tune_steps, reward_steps = [], [], [], []
            train_step = test_step = tune_step = reward_step = 0
            q_values_by_seed[seed_idx] = {'avg_pred_q_values': {}, 'avg_target_q_values': {}}

            for epoch in tqdm(range(epochs), desc='Epochs'):
                # ------- Training Phase -------
                bve_agent.train()
                for states, actions, rews, next_states, dones in loaders['train']:
                    states, actions = states.to(device), actions.to(device)
                    rews = rews.unsqueeze(1).to(device)
                    next_states = next_states.to(device)
                    dones = dones.unsqueeze(1).to(device)

                    loss, pred_q, target_q = bve_agent.learn((states, actions, rews, next_states, dones))

                    log_loss = np.log10(loss + 1e-8)
                    train_losses.append(log_loss)
                    train_steps.append(train_step)
                    bve_logger.log(f'{main_tag}/Train Loss', log_loss, train_step)
                    train_step += 1

                    q_values_by_seed[seed_idx]['avg_pred_q_values'][train_step] = [pred_q.mean().item()]
                    q_values_by_seed[seed_idx]['avg_target_q_values'][train_step] = [target_q.mean().item()]

                    free_up_memory([states, actions, rews, next_states, dones])
                
                # ---- Tune Phase ----
                bve_agent.eval()
                epoch_tune_losses = []
                with torch.no_grad():
                    for states, actions, rews, next_states, dones in loaders['tuning']:
                        states, actions = states.to(device), actions.to(device)
                        rews = rews.unsqueeze(1).to(device)
                        next_states = next_states.to(device)
                        dones = dones.unsqueeze(1).to(device)

                        next_value = bve_agent.target_q_net(next_states, bve_agent.get_action(next_states))
                        target = rews + bve_agent.gamma * (1 - dones) * next_value
                        current_q = bve_agent.q_net(states, actions)
                        loss = F.smooth_l1_loss(current_q, target)

                        log_loss = np.log10(loss.item() + 1e-8)
                        tune_losses.append(log_loss)
                        epoch_tune_losses.append(log_loss)
                        tune_steps.append(tune_step)
                        bve_logger.log(f'{main_tag}/Tune Loss', log_loss, tune_step)
                        tune_step += 1

                        free_up_memory([states, actions, rews, next_states, dones])
                
                # -- Learning rate decay based on tuning loss --
                if epoch_tune_losses:
                    mean_tune = np.mean(epoch_tune_losses)
                    for opt in [bve_agent.optimizer]:
                        for pg in opt.param_groups:
                            pg['lr'] *= 0.98 if mean_tune > 0.5 else 1.0
                
                # ------- Test Phase -------
                with torch.no_grad():
                    for states, actions, rews, next_states, dones in loaders['test']:
                        states, actions = states.to(device), actions.to(device)
                        rews = rews.unsqueeze(1).to(device)
                        next_states = next_states.to(device)
                        dones = dones.unsqueeze(1).to(device)

                        next_value = bve_agent.target_q_net(next_states, bve_agent.get_action(next_states))
                        target = rews + bve_agent.gamma * (1 - dones) * next_value
                        current_q = bve_agent.q_net(states, actions)
                        loss = F.smooth_l1_loss(current_q, target)

                        log_loss = np.log10(loss.item() + 1e-8)
                        test_losses.append(log_loss)
                        test_steps.append(test_step)
                        bve_logger.log(f'{main_tag}/Test Loss', log_loss, test_step)
                        test_step += 1

                        free_up_memory([states, actions, rews, next_states, dones])

                # ---- Reward Evaluation ----
                reward_list = evaluate_BVE_reward(bve_env, bve_agent, n_episodes=15, device=device, max_steps_per_episode=3000)
                for r in reward_list:
                    rewards.append(r)
                    reward_steps.append(reward_step)
                    bve_logger.log(f'{main_tag}/Reward', r, reward_step)
                    reward_step += 1

            # Save data
            all_train_losses.append(train_losses)
            all_tune_losses.append(tune_losses)
            all_test_losses.append(test_losses)
            all_rewards.append(rewards)
            all_train_steps.append(train_steps)
            all_tune_steps.append(tune_steps)
            all_test_steps.append(test_steps)
            all_reward_steps.append(reward_steps)

            print(f"Finished Training on {dataset_name}")
            print(f"    ➤ Avg Train Loss: {np.mean(train_losses):.5f}")
            print(f"    ➤ Avg Test Loss: {np.mean(test_losses):.5f}")
            print(f"    ➤ Avg Reward: {np.mean(rewards):.2f}")

            if seed_idx == 0:
                model_path = f"{logdir}/{dataset_type}/{perturbation_level}/bve_model_{perturbation_level}.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(bve_agent.state_dict(), model_path)
                print(f"Saved model to {model_path}")

            bve_env.close()
            time.sleep(1.5)

        stats_to_save = {
            'train_losses': all_train_losses,
            'test_losses': all_test_losses,
            'tune_losses': all_tune_losses,
            'rewards': all_rewards,
            'train_steps': all_train_steps,
            'test_steps': all_test_steps,
            'tune_steps': all_tune_steps,
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