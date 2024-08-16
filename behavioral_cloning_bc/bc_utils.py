# ----- PyTorch imports -----
import torch
import torch.nn as nn
import torch.optim as optim

# ----- Python imports -----
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_one_BC_epoch(model, loader, criterion, optimizer, device):
    """
    Function to train the BC model for one epoch.

    :param model: the BC model
    :param loader: the DataLoader object
    :param criterion: the loss function
    :param optimizer: the optimizer to use
    :param device: the device

    :return: the average loss for the epoch
    """
    model.train() # set model to training mode
    running_loss = 0.0

    for states, actions, _, _, _ in loader:
        # move states and actions to device
        states, actions = states.to(device), actions.to(device)

        # zero parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(states)
        loss = criterion(outputs, actions)

        # backward pass and optimize
        loss.backward()
        optimizer.step()

        # add the loss to the running loss
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
    """
    Evaluate the reward of the model in the environment.

    :param env: the environment
    :param model: the model
    :param n_episodes: the number of episodes to evaluate

    :return: the average reward over the episodes
    """
    rewards = []

    # loop through episodes
    for episode in range(n_episodes):
        observation = env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                # get the action from the model
                action = model(torch.tensor(observation, dtype=torch.float).to(device)).cpu().numpy()

            # take the action in the environment
            observation, reward, done, _ = env.step(action)
            total_reward += reward # add reward to total reward

            # check if episode is done, if so break
            if done:
                break

        # append total reward to rewards list
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

def train_and_evaluate_BC(dataloaders, loggerpath, device, trials, epochs, dataset):
    """
    Train and evaluate the BC model, logging loss and reward for each epoch.

    :param dataloaders: dictionary containing the dataloaders for each dataset
    :param loggerpath: the path for the logger
    :param device: the device
    :param trials: the number of trials
    :param epochs: the number of epochs
    :param dataset: the dataset to train on
    """

    # train and evaluate the BC model on all datasets
    for dataset_name, loaders in ((d, l) for d, l in dataloaders.items() if d == dataset):
        print(f"Training on {dataset_name}")

        # lists to store losses and rewards for plotting
        all_train_losses, all_test_losses, all_rewards = [], [], []

        # loop through trials
        for trial in range(trials):
            print(f"-- Starting Trial {trial + 1}/{trials} --")

            # ------- Setup Phase -------
            # logger setup
            bc_logger = TensorBoardLogger('bc_training_logs', dataset_path=loggerpath)

            # create the Enviornment
            bc_env, bc_action_dim, bc_state_dim = create_env(
                logger=bc_logger,
                env_id=ENV_ID,
                capture_video=True,
                seed=SEED,
                trial_number=f'{trial+1}'
                )

            # initialize the BC model, optimizer, loss function, and scheduler
            bc_model = BCModel(
                bc_state_dim,
                bc_action_dim,
                trial).to(device)
            optimizer = optim.Adam(bc_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, min_lr=1e-5)

            # print initial hyperparameters
            initial_lr = optimizer.param_groups[0]['lr']

            # lists to store losses and rewards for this trial
            train_losses, test_losses, rewards = [], [], []

            # loop through epochs
            for epoch in tqdm(range(epochs), desc='Epochs'):
                if epoch == 0:
                    print(f"Initial learning rate: {initial_lr:.6f}")

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
                    print(f"Adjusted learning rate in epoch {epoch + 1}: {adjusted_lr:.6f}")
                    initial_lr = adjusted_lr  # update for next comparison

                # ------- Test Phase -------
                test_loss = evaluate_BC(bc_model, criterion, loaders['test'], device)
                test_losses.append(test_loss)
                bc_logger.log(f'{main_tag}/Test Loss', test_loss, epoch + 1)

                # evaluation and reward collection at the end of each epoch
                epoch_reward = evaluate_reward(bc_env, bc_model, n_episodes=20)  # evaluate average over episodes per epoch
                rewards.extend(epoch_reward)
                bc_logger.log(f'{main_tag}/Epoch Reward', epoch_reward[0], epoch + 1) # log reward

            # collect losses and rewards for all trials
            all_train_losses.append(train_losses)
            all_test_losses.append(test_losses)
            all_rewards.append(rewards)

            # print final losses and rewards for this trial
            print(f"Finished Training on {dataset_name} - Training Loss: {train_loss:.5f}")
            print(f"Finished Tuning on {dataset_name} - Tuning Loss: {tune_loss:.5f}")
            print(f"Finished Testing on {dataset_name} - Test Loss: {test_loss:.5f}")
            print(f"Finished Evaluating on {dataset_name} - average Reward: {(sum(rewards) / len(rewards)):.2f}")

            # close the enviornment after each trial
            bc_env.close()

        print()

        # plot losses and rewards after all trials
        plot_losses_and_rewards(all_train_losses, all_test_losses, all_rewards, dataset_name, trials)

    # close the logger
    bc_logger.close()

