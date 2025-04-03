# utils.py

import gymnasium as gym
import os
import io
import base64
import pickle
from IPython.display import HTML, display
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics


def show_videos(video_data):
    """
    Function to display videos recorded during training with custom headers for each video, arranged side by side.

    :param video_data: list of tuples containing path and header title for each video
    """
    if not video_data:
        print("No video data provided.")
        return

    video_html = "<div style='display: flex; flex-wrap: wrap; align-items: flex-start;'>"

    # iterate over each video tuple in the list
    for video_path, header in video_data:
        try:
            with io.open(video_path, 'r+b') as f:
                video = f.read()
            encoded = base64.b64encode(video).decode('ascii')
            video_html += f'''
                <div style="margin: 5px; text-align: center;">
                    <h4 style="margin-bottom: 5px; color: white; background-color: #f57c00; padding: 10px; border-radius: 5px; font-size: 18px;">{header}</h4>
                    <video alt="{header}" autoplay loop controls style="height: 400px;">
                        <source src="data:video/mp4;base64,{encoded}" type="video/mp4" />
                    </video>
                </div>
            '''
        except IOError:
            print(f"Could not read video file: {video_path}")

    video_html += "</div>"

    # display HTML if any videos were found and encoded
    if video_html:
        display(HTML(data=video_html))
    else:
        print("Could not find or read any videos from the provided data.")


# Updated create_env function
def create_env(logger, seed, env_id='', capture_video=True, dataset_name='', trial_number=''):
    # Determine video directory
    video_dir = logger.get_video_dir(dataset_name, trial_number)
    os.makedirs(video_dir, exist_ok=True)

    # Create environment
    env = gym.make(env_id, render_mode='rgb_array')

    # Set render_fps
    if 'render_fps' not in env.metadata or env.metadata['render_fps'] is None:
        env.metadata['render_fps'] = 30  # Set desired FPS

    # Apply CustomRecordVideo before RecordEpisodeStatistics
    if capture_video:
        env = RecordVideo(env, video_dir, episode_trigger=lambda idx: True, disable_logger=True)
    
    # Apply other wrappers
    env = RecordEpisodeStatistics(env)

    # Set seed for reproducibility
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Diagnostics to understand action space
    #print("Action Space Type:", type(env.action_space))
    #print("Determined Action Size:", env.action_space.n)

    return env, env.action_space.n
    

def save_loss_curves(loss_data, filename):
    """
    Save the loss curves data to a file.
    
    :param loss_data: Dictionary containing the loss curves data.
    :param filename: Name of the file to save the loss data.
    """
    with open(filename, 'wb') as f:
        pickle.dump(loss_data, f)

def save_return_stats(data, filename):
    """
    Save the IQL return stats data to a file.
    
    :param loss_data: Dictionary containing the loss curves data.
    :param filename: Name of the file to save the loss data.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_loss_curves(filename):
    """
    Load the loss curves data from a file.
    
    :param filename: Name of the file to load the loss data.
    
    :return: Loaded loss data.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def load_iql_return_stats(filename):
    """
    Load the IQL return stats data from a file.
    
    :param filename: Name of the file to load the loss data.
    
    :return: Loaded loss data.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

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