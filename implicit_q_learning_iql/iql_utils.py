# ----- PyTorch imports -----
import torch
import torch.nn as nn
import torch.optim as optim

# ----- Python imports -----
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def adjust_learning_rate(optimizer, decay_factor):
    """
    Adjusts the learning rate of the optimizer by a specified decay factor.

    :param optimizer: optimizer whose learning rate needs adjustment
    :param decay_factor: factor by which to multiply the current learning rate
    """
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr'] # get current learning rate
        new_lr = old_lr * decay_factor
        param_group['lr'] = new_lr # set new learning rate

def tune_IQL(agent, data_loader):
    """
    Tunes the IQL agent using the provided data loader for the tuning dataset, calculating the mean squared error loss.

    :param agent: IQL agent being tuned
    :param data_loader: dataLoader containing tuning dataset

    :return: average loss computed over tuning dataset
    """
    total_loss = 0.0
    total_count = 0

    agent.eval()  # set agent to evaluation mode

    criterion = nn.MSELoss()  # define a loss function, e.g. MSE for action prediction

    with torch.no_grad():
        for states, true_actions, rewards, _, _ in data_loader:
            states = states.to(agent.device)
            true_actions = true_actions.to(agent.device)

            # predict actions using policy
            predicted_actions = agent.get_action(states, eval=True)

            # convert predicted actions to a tensor (if they are numpy arrays)
            if isinstance(predicted_actions, np.ndarray):
                predicted_actions = torch.tensor(predicted_actions).to(agent.device)

            # calculate MSE loss between predicted and true actions
            loss = criterion(predicted_actions, true_actions)

            # sum up rewards and loss
            total_loss += loss.item()
            total_count += len(rewards)

    # calculate average loss
    avg_loss = total_loss / total_count

    return avg_loss

def evaluate(agent, env, num_episodes):
    """
    Evaluates the IQL agent in a specified environment over a number of episodes.

    :param agent: IQL agent to evaluate
    :param env: environment in which to evaluate the agent
    :param num_episodes: number of episodes to run for evaluation

    :return: dictionary containing average reward and episode length
    """
    stats = {'reward': [], 'length': []}

    # loop through episodes
    for _ in range(num_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        # get action, take action, and update statistics
        while not done:
            action = agent.get_action(observation, eval=True)
            observation, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_length += 1

        stats['reward'].append(episode_reward)
        stats['length'].append(episode_length)

    # calculate and return average statistics
    stats = {k: np.mean(v) for k, v in stats.items()}

    return stats

def train_one_IQL_epoch(agent, loader):
    """
    Conducts one epoch of training for an IQL agent using the provided data loader.

    :param agent: The IQL agent to train
    :param loader: DataLoader containing the training data

    :return: A tuple containing the average losses and Q-values
    """
    # initialize lists to store losses and Q-values
    policy_losses, critic1_losses, critic2_losses, value_losses = [], [], [], []
    avg_pred_q_values, avg_target_q_values = [], []

    agent.train() # set agent to training mode

    # loop through the data loader
    for states, actions, rewards, next_states, dones in loader:
        states = states.to(agent.device)
        actions = actions.to(agent.device)
        rewards = rewards.unsqueeze(1).to(agent.device)
        next_states = next_states.to(agent.device)
        dones = dones.unsqueeze(1).to(agent.device)

        # update agent and get losses
        policy_loss, critic1_loss, critic1_pred_q, critic1_target_q, critic2_loss, critic2_pred_q, critic2_target_q, value_loss = agent.learn((states, actions, rewards, next_states, dones))

        # add losses to lists
        policy_losses.append(policy_loss)
        critic1_losses.append(critic1_loss)
        critic2_losses.append(critic2_loss)
        value_losses.append(value_loss)

        # add Q-values to lists
        avg_pred_q_values.append((critic1_pred_q.mean().item() + critic2_pred_q.mean().item()) / 2)
        avg_target_q_values.append((critic1_target_q.mean().item() + critic2_target_q.mean().item()) / 2)

    # at the end of the epoch, print the average Q-values
    #print(f"Average predicted Q-value: {np.mean(avg_pred_q_values):.4f}")
    #print(f"Average target Q-value: {np.mean(avg_target_q_values):.4f}")

    return np.mean(policy_losses), np.mean(critic1_losses), np.mean(critic2_losses), np.mean(value_losses), avg_pred_q_values, avg_target_q_values

def train_iql(dataloaders, epochs, trials, dataset, loggerpath):
    """
    Trains an Implicit Q-Learning (IQL) agent on the specified dataset.

    :param dataloaders: dictionary containing training, validation, and test data loaders
    :param epochs: number of training epochs
    :param trials: number of training trials
    :param dataset: name of the dataset being used
    :param loggerpath: the path for the logger
    """
    stats_to_return = {}

    # init dictionary to store Q-values by epoch
    epoch_q_values = {
        trial: {
            'avg_pred_q_values': {epoch: [] for epoch in range(epochs)},
            'avg_target_q_values': {epoch: [] for epoch in range(epochs)}
        } for trial in range(trials)
    }

    # loop through datasets
    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training IQL on {dataset_name}")

        # lists to store losses and rewards for plotting
        all_policy_losses, all_critic1_losses, all_critic2_losses, all_value_losses, all_rewards = [], [], [], [], []

        # loop through trials
        for trial in range(trials):
            print(f"-- Starting Trial {trial + 1}/{trials} --")

            # ------- Setup Phase -------
            # logger setup
            iql_logger = TensorBoardLogger('iql_training_logs', dataset_path=loggerpath)

            # create the Enviornment
            iql_env, iql_action_dim, iql_state_dim = create_env(
                logger=iql_logger,
                env_id=ENV_ID,
                capture_video=True,
                seed=SEED,
                trial_number=f'{trial+1}'
                )

            # reinitialize the IQL model for each dataset
            iql_agent = IQLAgent(state_size=iql_state_dim,
                                action_size=iql_action_dim,
                                learning_rate=3e-4,
                                hidden_size=256,
                                tau=5e-3,
                                temperature=0.1,
                                expectile=0.7,
                                device=device,
                                trial_idx=trial
                                )

            # initialize optimizers with learning rates and possibly other hyperparameters
            actor_optimizer = optim.Adam(iql_agent.actor_network.parameters(), lr=0.001)
            critic_optimizer = optim.Adam(list(iql_agent.critic1_network.parameters()) + list(iql_agent.critic2_network.parameters()), lr=0.001)
            value_optimizer = optim.Adam(iql_agent.value_network.parameters(), lr=0.001)

            # lists to store losses and rewards for this trial
            policy_losses, critic1_losses, critic2_losses, value_losses, rewards = [], [], [], [], []

            # Hyperparameter Tuning - Initialize variables to track best tuning loss and patience counter
            best_tune_loss = float('inf')
            no_improvement_epochs = 0
            lr_decay_factor = 0.9
            patience = 3

            # loop through epochs
            for epoch in tqdm(range(epochs), desc='Epochs'):
                # ------- Initial Training Phase -------
                iql_agent.train()
                policy_loss, critic1_loss, critic2_loss, value_loss, avg_pred_q_values, avg_target_q_values = train_one_IQL_epoch(iql_agent, loaders['train'])

                # collect losses for plotting
                policy_losses.append(policy_loss)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
                value_losses.append(value_loss)

                # store Q-values for current epoch and trial
                epoch_q_values[trial]['avg_pred_q_values'][epoch] = avg_pred_q_values
                epoch_q_values[trial]['avg_target_q_values'][epoch] = avg_target_q_values

                # log training loss
                main_tag = f'{dataset_name}_Trial_{trial + 1}'
                iql_logger.log(f'{main_tag}/Poicy Loss', policy_loss, epoch + 1)
                iql_logger.log(f'{main_tag}/Critc1 Loss', critic1_loss, epoch + 1)
                iql_logger.log(f'{main_tag}/Critc2 Loss', critic2_loss, epoch + 1)
                iql_logger.log(f'{main_tag}/Value Loss', value_loss, epoch + 1)

                # ------- Hyperparameter Tuning Phase -------
                tune_loss = tune_IQL(iql_agent, loaders['tuning'])
                #iql_logger.log(f'{main_tag}/Tune Loss', tune_loss, epoch + 1)

                # check if there is an improvement
                if tune_loss < best_tune_loss:
                    best_tune_loss = tune_loss
                    no_improvement_epochs = 0  # reset counter
                else:
                    no_improvement_epochs += 1

                # if no improvement equal to patience, reduce learning rate
                if no_improvement_epochs >= patience:
                    no_improvement_epochs = 0  # reset counter

                    # reduce learning rate
                    adjust_learning_rate(actor_optimizer, lr_decay_factor)
                    adjust_learning_rate(critic_optimizer, lr_decay_factor)
                    adjust_learning_rate(value_optimizer, lr_decay_factor)
                    print(f"Learning rate reduced by {lr_decay_factor} at epoch {epoch+1}")


                # ------- Test Phase on Test Set -------
                stats = evaluate(iql_agent, iql_env, num_episodes=20)
                rewards.append(stats['reward'])
                #print("Evaluation Stats:", stats)
                iql_logger.log(f'{main_tag}/Reward', stats['reward'], epoch + 1)

            # collect losses for all trials
            all_policy_losses.append(policy_losses)
            all_critic1_losses.append(critic1_losses)
            all_critic2_losses.append(critic2_losses)
            all_value_losses.append(value_losses)
            all_rewards.append(rewards)

            # print final loss values for this trial
            print(f"Finished Training on {dataset_name} - Policy Loss: {policy_loss:.5f}")
            print(f"                                    - Critic1 Loss: {critic1_loss:.5f}")
            print(f"                                    - Critic2 Loss: {critic2_loss:.5f}")
            print(f"                                    - Value Loss: {value_loss:.5f}")

            # close the enviornment after each trial
            iql_env.close()

        stats_to_return = {
            'policy_losses': all_policy_losses,
            'critic1_losses': all_critic1_losses,
            'critic2_losses': all_critic2_losses,
            'value_losses': all_value_losses,
            'rewards': all_rewards,
            'q_values': epoch_q_values,
            'dataset_name': dataset_name,
            'trials': trials
        }

        print()

    # close TensorBoard logger
    iql_logger.close()

    return stats_to_return

def plot_iql_losses_and_rewards(policy_losses, critic1_losses, critic2_losses, value_losses, rewards, dataset_name, trials):
    """
    Plot the policy, critic, value losses and rewards from IQL training.

    :param policy_losses: list of policy losses for each trial
    :param critic1_losses: list of critic 1 losses for each trial
    :param critic2_losses: list of critic 2 losses for each trial
    :param value_losses: list of value losses for each trial
    :param rewards: list of rewards for each trial
    :param dataset_name: name of the dataset
    :param trials: number of trials
    """
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    axs[0].set_title(f'Policy Loss - {dataset_name}')
    axs[1].set_title(f'Critic 1 Loss - {dataset_name}')
    axs[2].set_title(f'Critic 2 Loss - {dataset_name}')
    axs[3].set_title(f'Value Loss - {dataset_name}')
    axs[4].set_title(f'Rewards per Epoch - {dataset_name}')

    # plot losses and rewards for each trial
    for trial in range(trials):
        axs[0].plot(policy_losses[trial], label=f'Trial {trial+1}')
        axs[1].plot(critic1_losses[trial], label=f'Trial {trial+1}')
        axs[2].plot(critic2_losses[trial], label=f'Trial {trial+1}')
        axs[3].plot(value_losses[trial], label=f'Trial {trial+1}')
        axs[4].plot(rewards[trial], label=f'Trial {trial+1}')

    for ax in axs:
        ax.set_xlabel('Epoch')
        ax.legend()

    axs[0].set_ylabel('Loss')
    axs[4].set_ylabel('Total Reward')  # set y label for rewards plot

    plt.tight_layout()
    plt.show()

def calculate_similarity(q_values_dict):
    """
    Calculates the cosine similarity between predicted and target Q-values for each trial and epoch.

    :param q_values_dict: dictionary containing Q-values for each trial and epoch
    :return: matrix of similarities and DataFrame of percentage similarities
    """
    trials = len(q_values_dict)  # get number of trials
    epochs = len(q_values_dict[0]['avg_pred_q_values'])  # get number of epochs

    similarities = np.zeros((trials, epochs))  # initialize array to store similarities

    for trial in range(trials):
        for epoch in range(epochs):
            # calculate cosine similarity between predicted and target Q-values for each epoch and trial
            pred_q_values = np.array(q_values_dict[trial]['avg_pred_q_values'][epoch])
            target_q_values = np.array(q_values_dict[trial]['avg_target_q_values'][epoch])
            similarity = cosine_similarity([pred_q_values], [target_q_values])[0][0]
            similarities[trial][epoch] = similarity

    # convert similarity matrix to DataFrame with trial and epoch as row and column names
    similarity_df = pd.DataFrame(similarities, index=['Trial {}'.format(i+1) for i in range(trials)], columns=['Epoch {}'.format(i+1) for i in range(epochs)])

    # calculate percentage similarity
    percentage_similarity = similarity_df * 100

    return similarities, percentage_similarity

def plot_q_value_similarity(q_values_dict):
    """
    Plots the cosine similarity between predicted and target Q-values for each trial and epoch.

    :param q_values_dict: dictionary containing Q-values for each trial and epoch
    """
    # check if there are enough trials to plot first and last
    if len(q_values_dict) < 2:
        raise ValueError("Not enough trials to plot both first and last.")

    # select first and last trial
    selected_trials = [0, len(q_values_dict) - 1]
    epochs = len(q_values_dict[0]['avg_pred_q_values'])

    # create subplots for first and last trials
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))  # Two rows, one for each trial

    # flatten axes array for easy indexing
    axes = axes.flatten()

    # iterate through selected trials (first and last)
    for idx, trial_index in enumerate(selected_trials):
        trial_data = q_values_dict[trial_index]

        # --- Plotting the first epoch ---
        pred_q_values_first = trial_data['avg_pred_q_values'][0]
        target_q_values_first = trial_data['avg_target_q_values'][0]

        axes[idx*2].plot(pred_q_values_first, label='Predicted Q-values')
        axes[idx*2].plot(target_q_values_first, label='Target Q-values', alpha=0.5)
        axes[idx*2].set_title(f'Trial {trial_index + 1}: First Epoch')
        axes[idx*2].legend()

        # --- Plotting the last epoch ---
        pred_q_values_last = trial_data['avg_pred_q_values'][epochs - 1]
        target_q_values_last = trial_data['avg_target_q_values'][epochs - 1]

        axes[idx*2+1].plot(pred_q_values_last, label='Predicted Q-values')
        axes[idx*2+1].plot(target_q_values_last, label='Target Q-values', alpha=0.5)
        axes[idx*2+1].set_title(f'Trial {trial_index + 1}: Last Epoch')
        axes[idx*2+1].legend()

    # add X and Y labels
    fig.text(0.5, -0.01, 'Index within Epoch', ha='center', va='center')
    fig.text(0.01, 0.5, 'Q-value', ha='center', va='center', rotation='vertical')

    plt.tight_layout()
    plt.show()
