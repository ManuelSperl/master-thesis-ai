# utils.py

import gymnasium as gym
import torch
import gc
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
def create_env(logger, seed, env_id='', capture_video=True, dataset_name='', seed_number=''):
    # Determine video directory
    video_dir = logger.get_video_dir(dataset_name, seed_number)
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


def save_return_stats(data, filename):
    """
    Save the IQL return stats data to a file.
    
    :param loss_data: Dictionary containing the loss curves data.
    :param filename: Name of the file to save the loss data.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def free_up_memory(used_vars):
    """Free up memory by deleting specified variables."""
    for var in used_vars:
        del var
    torch.cuda.empty_cache()
    gc.collect()