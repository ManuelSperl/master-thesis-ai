# Master’s Thesis: Evaluating Offline Reinforcement Learning Under Suboptimal Data Quality

This repository contains the full implementation of my master's thesis:  
**"Evaluating Offline Reinforcement Learning Under Suboptimal Data Quality: A Comparative Study of BC, BVE, and IQL."**

The project investigates how different offline reinforcement learning (RL) algorithms perform when trained on datasets with varying expert quality and perturbation levels. The study compares **Behavioral Cloning (BC)**, **Behavior Value Estimation (BVE)**, and **Implicit Q-Learning (IQL)** on the Atari game **[Seaquest](https://ale.farama.org/environments/seaquest/)**.

---

## Project Structure

```
master-thesis-ai-main/
│
├── main_datasets.ipynb           # Script for generating datasets with quality and perturbation control
├── main_training.ipynb           # Main notebook to train BC, BVE, and IQL models
├── main_plotting.ipynb           # Notebook to visualize training results and dataset statistics
│
├── auxiliary_methods/            # Helper functions and logging utilities
│   ├── utils.py                  
│   └── tb_logger.py              
│
├── offline_rl_models/            # Model-specific implementations
│   ├── behavioral_cloning_bc/    # BC model and training logic
│   ├── behavior_value_estimation_bve/  # BVE method based on IQL Equation (2)
│   └── implicit_q_learning_iql/  # Full IQL implementation (actor, critic, value network)
│
├── plotting/                     # Helper functions for plotting
│   └── plot_utils.py
│
├── datasets/                     # Dataset helper functions
│   ├── dataset_utils.py
│   └── seaquest_dataset.py
│
├── .vscode/                      # VSCode settings (optional)
├── .gitignore                    # Ignore logs, checkpoints, and cached files
└── README.md                     # Project overview (you're reading it!)
```

---

## Key Concepts

- **Offline RL**: Training agents from static datasets without environment interaction.
- **Suboptimal Data Quality**: Datasets collected from agents with different skill levels (beginner, intermediate, expert) and action perturbations (0% to 20%).
- **Methods Compared**:
  - **BC (Behavioral Cloning)** [1]: Supervised learning on state-action pairs.
  - **BVE (Behavior Value Estimation)**: Value learning using fixed next-actions from the dataset (based on IQL Eq. 2).
  - **IQL (Implicit Q-Learning)** [2]: TD-based critic and expectile value learning with advantage-weighted regression.

---

## How to Run Experiments

### 1. Generate Offline Datasets
```bash
# Open and run
main_datasets.ipynb
```
You can control the expert type (`beginner`, `intermediate`, `expert`) and perturbation level (0%, 5%, 10%, 20%).

---

### 2. Train Offline RL Models
```bash
# Open and run
main_training.ipynb
```
This will train all models (BC, BVE, IQL) on all dataset variants, logging performance over time.

---

### 3. Visualize Results
```bash
# Open and run
main_plotting.ipynb
```
Generates dot plots of action distributions, perturbation heatmaps, and learning curves (losses and rewards).

---

## Logging and Visualization

All training logs (losses, rewards) are stored using **TensorBoard** and also saved to `.pkl` for later plotting. Visualizations are created using `matplotlib` and `seaborn`.

---

## Environment & Dependencies

- Python 3.8+
- Gymnasium (Seaquest environment)
- PyTorch
- Stable Baselines3
- Seaborn, Matplotlib
- NumPy, Pickle, tqdm

Install requirements:
```bash
pip install -r requirements.txt
```
(*You may need to manually add `requirements.txt` with your environment packages if not included.*)

---

## Environment & Dependencies

To ensure full reproducibility, this project provides a complete Conda environment file.

### Option 1: Recreate the Conda Environment (Recommended)
```bash
conda env create -f environment.yml
conda activate master-thesis-ai
```

> This will install all required packages and versions exactly as used in the thesis experiments.

---

### Option 2: Install with pip (alternative, not guaranteed to match exact versions)
If you prefer using `pip`, and a `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

---

### Main Dependencies

- Python 3.8+
- [Gymnasium](https://gymnasium.farama.org/) (Atari / Seaquest)
- PyTorch
- Stable Baselines3
- Matplotlib, Seaborn
- NumPy, tqdm, Pickle

---

## Thesis Highlights

- Demonstrates **BC's robustness** even under suboptimal data.
- Shows **IQL's instability** when faced with noisy or perturbed actions.
- Introduces **BVE** as a hybrid value estimation method grounded in IQL theory but simpler to train.

---

## Citation

If you use this code or findings in your work, please cite:

> [Manuel Sperl]. (2025). *Evaluating Offline Reinforcement Learning Under Suboptimal Data Quality: A Comparative Study of BC, BVE, and IQL.* Master’s Thesis, [Johannes Kepler Universität, JKU].

---

## References

[1] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu (2020).  
*Offline Reinforcement Learning: Tutorial, Review, and Perspectives.*  
[arXiv:2005.01643](https://arxiv.org/abs/2005.01643)

[2] Ilya Kostrikov, Ashvin Nair, Sergey Levine (2022).  
*Offline Reinforcement Learning with Implicit Q-Learning.*  
[arXiv:2110.06169](https://arxiv.org/abs/2110.06169)
