# ----- PyTorch imports -----
import torch
import torch.nn as nn
import torch.optim as optim

# ----- Python imports -----
from tqdm import tqdm
import matplotlib.pyplot as plt

import importlib

import auxiliary_methods.tb_logger
importlib.reload(auxiliary_methods.tb_logger)
from auxiliary_methods.tb_logger import TensorBoardLogger

import auxiliary_methods.utils
importlib.reload(auxiliary_methods.utils)
from auxiliary_methods.utils import create_env
from auxiliary_methods.utils import save_loss_curves

import behavioral_cloning_bc.bc_model
importlib.reload(behavioral_cloning_bc.bc_model)
from behavioral_cloning_bc.bc_model import BCModel

def train_one_BC_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for states, actions, _, _, _ in loader:
        states, actions = states.to(device), actions.to(device)

        optimizer.zero_grad()

        outputs = model(states)  # Outputs are logits
        loss = criterion(outputs, actions)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss

def evaluate_BC(model, criterion, data_loader, device):
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

    with torch.no_grad():
        for (states, actions, _, _, _) in data_loader:
            # move states and actions to device
            states, actions = states.to(device), actions.to(device)

            # forward pass
            outputs = model(states)
            loss = criterion(outputs, actions)

            # add the loss to the total loss
            total_loss += loss.item()

    eval_loss = total_loss / len(data_loader)
    return eval_loss

def evaluate_reward(env, model, n_episodes, device):
    rewards = []

    for episode in range(n_episodes):
        observation, info = env.reset()

        # Set frames_per_sec if the video recorder exists
        video_recorder = env.get_wrapper_attr('video_recorder')
        if video_recorder is not None:
            video_recorder.frames_per_sec = 30

        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                # Process observation
                state = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                logits = model(state)
                action = torch.argmax(logits, dim=1).item()

            # Take the action in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Set frames_per_sec after the first step (if needed)
            if video_recorder is not None:
                video_recorder.frames_per_sec = 30

        rewards.append(total_reward)

    return [sum(rewards) / len(rewards)]


def plot_losses_and_rewards(train_losses, test_losses, rewards, dataset_name, trials):
    """
    Plot training, testing losses and rewards.

    :param train_losses: list of training losses
    :param test_losses: list of testing losses
    :param rewards: list of rewards
    :param dataset_name: name of the dataset
    :param trials: number of trials
    """
    # set up the plot, with corresponding titles
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].set_title(f'Training Loss - {dataset_name}')
    ax[1].set_title(f'Test Loss - {dataset_name}')
    ax[2].set_title(f'Rewards per Epoch - {dataset_name}')

    # loop through trials and plot the losses and rewards
    for trial in range(trials):
        ax[0].plot(train_losses[trial], label=f'Trial {trial+1}')
        ax[1].plot(test_losses[trial], label=f'Trial {trial+1}')
        ax[2].plot(rewards[trial], label=f'Trial {trial+1}') # plot rewards per epoch

    for a in ax:
        a.set_xlabel('Epoch')
        a.legend()

    ax[0].set_ylabel('Loss')
    ax[2].set_ylabel('Total Reward')  # set y label for rewards plot

    plt.tight_layout()
    plt.show()

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

    # train and evaluate the BC model on all datasets
    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training on {dataset_name}")

        # Split the key to extract dataset type and perturbation level
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
            bc_logger = TensorBoardLogger(
                            logdir=logdir,
                            dataset_type=dataset_type,
                            perturbation_level=perturbation_level
                        )

            # create the Enviornment
            bc_env, bc_action_dim = create_env(
                logger=bc_logger,
                env_id=env_id,
                capture_video=True,
                seed=seed,
                trial_number=f'{trial+1}'
                )

            # initialize the BC model, optimizer, loss function, and scheduler
            bc_model = BCModel(
                bc_action_dim,
                seed,
                trial
            ).to(device)
            optimizer = optim.Adam(bc_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, min_lr=1e-5)

            # print initial hyperparameters
            initial_lr = optimizer.param_groups[0]['lr']

            # lists to store losses and rewards for this trial
            train_losses, test_losses, rewards = [], [], []

            lr_updates = []  # To store any learning rate updates
            lr_updates.append(f"Initial learning rate: {initial_lr:.6f}")

            # loop through epochs
            for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
                # ------- Training Phase -------
                bc_model.train()
                train_loss = train_one_BC_epoch(bc_model, loaders['train'], criterion, optimizer, device)
                train_losses.append(train_loss)

                # log training loss
                main_tag = f'{dataset_name}_Trial_{trial + 1}'
                bc_logger.log(f'{main_tag}/Train Loss', train_loss, epoch + 1)

                # ------- Hyperparameter Tuning Phase -------
                tune_loss = evaluate_BC(bc_model, criterion, loaders['tuning'], device)
                bc_logger.log(f'{main_tag}/Tune Loss', tune_loss, epoch + 1)

                # adjust learning rate based on tuning loss
                scheduler.step(tune_loss)

                # print adjusted hyperparameters
                adjusted_lr = optimizer.param_groups[0]['lr']
                if adjusted_lr != initial_lr:
                    lr_updates.append(f"Adjusted learning rate in epoch {epoch + 1}: {adjusted_lr:.6f}")
                    initial_lr = adjusted_lr  # update for next comparison

                # ------- Test Phase -------
                test_loss = evaluate_BC(bc_model, criterion, loaders['test'], device)
                test_losses.append(test_loss)
                bc_logger.log(f'{main_tag}/Test Loss', test_loss, epoch + 1)

                # evaluation and reward collection at the end of each epoch
                epoch_reward = evaluate_reward(bc_env, bc_model, n_episodes=20, device=device)  # evaluate average over episodes per epoch
                rewards.extend(epoch_reward)
                bc_logger.log(f'{main_tag}/Epoch Reward', epoch_reward[0], epoch + 1) # log reward
            
            # Print all learning rate updates after epochs are completed
            for msg in lr_updates:
                print(msg)

            # collect losses and rewards for all trials
            all_train_losses.append(train_losses)
            all_test_losses.append(test_losses)
            all_rewards.append(rewards)

            # print final losses and rewards for this trial
            print(f"Finished Training on {dataset_name} - Training Loss: {train_loss:.5f}")
            print(f"Finished Tuning on {dataset_name} - Tuning Loss: {tune_loss:.5f}")
            print(f"Finished Testing on {dataset_name} - Test Loss: {test_loss:.5f}")
            print(f"Finished Evaluating on {dataset_name} - average Reward: {(sum(rewards) / len(rewards)):.2f}")

            # Save the loss curves and model after the first trial
            if trial == 0:
                loss_data = {
                    'train_losses': all_train_losses,
                    'test_losses': all_test_losses,
                    'rewards': all_rewards,
                }
                loss_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/loss_curves_{perturbation_level}.pkl"
                save_loss_curves(loss_data, loss_save_path)
                print(f"Loss curves saved to {loss_save_path}")

                model_save_path = f"{logdir}/{dataset_type}/perturbation_{perturbation_level}/bc_model_{perturbation_level}.pth"
                torch.save(bc_model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

            # close the enviornment after each trial
            bc_env.close()

        print()

        # plot losses and rewards after all trials
        plot_losses_and_rewards(all_train_losses, all_test_losses, all_rewards, dataset_name, trials)

    # close the logger
    bc_logger.close()

