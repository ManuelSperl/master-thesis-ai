# tb_logger.py

import os
import shutil
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """
    A logger class for logging TensorBoard data with dataset and perturbation level in the directory structure.
    """

    def __init__(self, logdir, dataset_type, perturbation_level, params=None):
        """
        Initialize the TensorBoard logger.

        :param logdir: Base directory for logs.
        :param dataset_type: Type of dataset.
        :param perturbation_level: Level of perturbation.
        :param params: Optional path to a parameters file to copy into the log directory.
        """
        # create the desired directory structure
        self.basepath = os.path.join(logdir, dataset_type, f"{perturbation_level}")
        self.log_dir = os.path.join(self.basepath, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # initialize TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        self.log_dict = {}

        # copy parameters file if provided
        if params is not None and os.path.exists(params):
            shutil.copyfile(params, os.path.join(self.basepath, "params.pkl"))

    def log(self, name, value, step=None):
        """
        Log values and optionally include them in TensorBoard.

        :param name: Name of the metric to log.
        :param value: Value of the metric to log.
        :param step: Optional step number for logging in TensorBoard.
        """
        if name not in self.log_dict:
            self.log_dict[name] = []

        self.log_dict[name].append(value)

        # log to TensorBoard if a step is provided
        if step is not None:
            self.writer.add_scalar(name, value, step)

    def get_video_dir(self, dataset_name='', seed_number=''):
        """
        Return the directory for saving videos, including dataset type and seed number in the path.

        :param dataset_name: Name of the dataset.
        :param seed_number: Seed number for the dataset.
        """
        seed_str = f"seed_{seed_number}" if seed_number else ""
        video_dir = os.path.join(self.basepath, "videos", dataset_name, seed_str)
        os.makedirs(video_dir, exist_ok=True)
        return video_dir

    def close(self):
        """
        Close the TensorBoard writer.
        """
        self.writer.close()
