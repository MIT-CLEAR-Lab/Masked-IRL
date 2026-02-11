# Masked IRL: LLM-Guided Reward Disambiguation from Demonstrations and Language

Official codebase for **Masked Inverse Reinforcement Learning (Masked IRL)**, a framework that uses large language models (LLMs) to combine demonstrations and natural language instructions for reward learning.

**Accepted to ICRA 2026**

**Paper**: [arXiv:2511.14565](https://arxiv.org/abs/2511.14565v1)  
**Project Page**: https://github.com/MIT-CLEAR-Lab/Masked-IRL

## Overview

Masked IRL addresses the fundamental ambiguity in inverse reinforcement learning by:
1. Disambiguating ambiguous instructions using demonstration context
2. Using LLMs to infer state-relevance masks from language instructions
3. Enforcing invariance to irrelevant state components through a masking loss

This enables more sample-efficient and generalizable reward learning compared to prior language-conditioned IRL methods.

## Installation

### Dependencies

1. Create a conda environment:
```bash
conda create -n maskedirl python=3.10
conda activate maskedirl
```

2. Install PyTorch (CUDA 12.4):
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

3. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

### Dataset Files

The codebase uses trajectory datasets stored in `data/traj_sets/`:

- **Simulation datasets**:
  - `frankarobot_obj20_sg10_persg5.npy` - Main simulation dataset (20 objects, 10 start-goal pairs, 5 trajectories per pair)
  - `frankarobot_obj20_sg10_persg5_shortest_paths.npy` - Shortest path trajectories for baseline comparison on extracting state relevance masks

- **Real robot datasets**:
  - `frankarobot_obj100_sg50_persg50.npy` - Real robot dataset (100 objects, 50 start-goal pairs, 50 trajectories per pair)
  - `frankarobot_obj100_sg50_persg50_dict.npy` - Dictionary format of real robot data

### Generating Trajectories

To generate new trajectory datasets:

```bash
# For simulation (set real: False in config)
PYTHONPATH=. python src/scripts/generate_traj_sets.py \
    --config config/trajset_gen/frankarobot.yaml \
    --samples 10000 \
    --seed 12345

# For real robot (set real: True in config to use GUI)
PYTHONPATH=. python src/scripts/generate_traj_sets.py \
    --config config/trajset_gen/frankarobot.yaml \
    --samples 1000 \
    --seed 12345
```

**Generating trajectories for multiple object locations**:

To generate trajectories with varying object locations:

```bash
PYTHONPATH=. python src/scripts/generate_trajs_obj_multiple_locations.py \
    --config config/trajset_gen/frankarobot.yaml \
    --num_obj_locations 20 \
    --samples 50 \
    --seed 0 \
    --save_dir ./data/traj_sets \
    --tag obj20_sg50_persg5
```

This script:
- Generates trajectories for multiple object location configurations
- Randomly samples `human_center`, `laptop_center`, and `table_center` for each location
- Combines all trajectories into a single array (or saves as dictionary with `--no-combine`)
- Uses `generate_traj_sets.py` internally for each location

### Human Preference Data

Human preference configurations are stored in `config/humans/`:
- `frankarobot_multiple_humans.yaml` - Main simulation human preferences
- `frankarobot_multiple_humans_validfeat1and2.yaml` - Real robot human preferences
- `thetas_sampled_data_simulation.json` - Simulation theta (preference) samples
- `thetas_sampled_data_realrobot.json` - Real robot theta samples

### Data Split Configuration

Data split configurations are stored in `config/data_split_config/`:
- `frankarobot_obj20_sg10_persg5/` - Simulation data splits
- `frankarobot_real/` - Real robot data splits

## Training

The codebase uses unified training scripts that support both simulation and real robot experiments. Use the `--realrobot` flag to switch between modes.

### Simulation Experiments

Train Masked IRL on simulation data (default mode):

```bash
# Direct Python execution
python src/scripts/train.py \
    --seed 12345 \
    --config config/reward_learning/obj20_sg10_persg5/maskedrl_llm_mask.yaml \
    -hc config/humans/frankarobot_multiple_humans.yaml \
    -dq 10 \
    --num_train_thetas 34 \
    --state_dim 19 \
    --lr 0.0001 \
    --batch_size 128 \
    --num_iterations 1000 \
    --language_ambiguity omit_expression \
    --llm_disambiguation llm \
    --wandb
```

**Using bash script wrapper**:

```bash
bash src/scripts/train.sh \
    -t obj20_sg10_persg5 \
    -m maskedrl_llm_mask \
    -s 12345 \
    -d 10 \
    -n 34 \
    -S 19 \
    -l 0.0001 \
    -b 128 \
    -I 1000 \
    -A omit_expression \
    -D llm \
    -w 1
```

**Using SLURM (sbatch)**:

```bash
# Submit job using sbatch
sbatch src/scripts/slurm.sbatch \
    -t obj20_sg10_persg5 \
    -m maskedrl_llm_mask \
    -s 12345 \
    -d 10 \
    -n 34 \
    -S 19 \
    -l 0.0001 \
    -b 128 \
    -I 1000 \
    -A omit_expression \
    -D llm \
    -w 1
```

### Real Robot Experiments

Train Masked IRL on real robot data using the `--realrobot` flag:

```bash
# Direct Python execution
python src/scripts/train.py \
    --realrobot \
    --seed 12345 \
    --config config/reward_learning/realrobot/maskedrl_llm_mask.yaml \
    -hc config/humans/frankarobot_multiple_humans_validfeat1and2.yaml \
    -dq 10 \
    --num_train_thetas 34 \
    --state_dim 19 \
    --lr 0.0001 \
    --batch_size 128 \
    --num_iterations 1000 \
    --wandb
```

**Using bash script wrapper**:

```bash
bash src/scripts/train.sh \
    -r \
    -t realrobot \
    -m maskedrl_llm_mask \
    -s 12345 \
    -d 10 \
    -n 34 \
    -S 19 \
    -l 0.0001 \
    -b 128 \
    -I 1000 \
    -w 1
```

**Note**: When using `-r` flag, the script automatically uses `realrobot` as the trajectory info, so you can also use `-t realrobot` explicitly.

**Note**: Real robot mode automatically:
- Uses different default human config files (`frankarobot_multiple_humans_validfeat1and2.yaml`)
- Uses `realrobot` trajectory info (configs in `config/reward_learning/realrobot/`)
- Requires existing split indices from `config/data_split_config/frankarobot_real/` (no auto-splitting)
- Disables LLM disambiguation (not supported for real robot)
- Uses different test trajectory paths

### Training Parameters

Key parameters for training:

- `--seed`: Random seed for reproducibility
- `--config`: Path to reward learning config YAML file
- `-hc, --human_config`: Path to human preference config YAML file (defaults differ for simulation vs real robot)
- `-dq, --demo_queries`: Number of demonstration queries per preference
- `--num_train_thetas`: Number of training preferences (thetas)
- `--state_dim`: State dimension (19 for full state, 9 for end-effector only)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 512)
- `--num_iterations`: Number of training iterations (default: 100)
- `--language_ambiguity`: Type of ambiguity (`omit_expression`, `omit_referent`, `paraphrase`, or `None`)
- `--llm_disambiguation`: Enable LLM disambiguation (`llm`, `vlm`, or `None`) - simulation only
- `--realrobot`: Use real robot mode (different defaults and behavior)
- `--wandb`: Enable Weights & Biases logging

### Model Configurations

Config files are organized by dataset type in `config/reward_learning/`:

- **Simulation configs** (`obj20_sg10_persg5/`):
  - `maskedrl.yaml` - Masked IRL (base)
  - `maskedrl_llm_mask.yaml` - Masked IRL with LLM masks
  - `maskedrl_no_lang.yaml` - Masked IRL without language
  - `explicitmask.yaml` - Explicit mask baseline
  - `explicitmask_llm_mask.yaml` - Explicit mask with LLM masks
  - `explicitmask_no_lang.yaml` - Explicit mask without language
  - `meirl.yaml` - Maximum Entropy IRL
  - `meirl_no_lang.yaml` - Maximum Entropy IRL without language
  - `multiple_maskedrl.yaml` - Masked IRL for multiple objects (dictionary format)
  - `multiple_explicitmask.yaml` - Explicit mask for multiple objects
  - `multiple_meirl.yaml` - Maximum Entropy IRL for multiple objects

- **Real robot configs** (`realrobot/`):
  - Same naming convention as simulation configs (without `multiple_*` variants)

Key config parameters:
- `masked_loss_weight`: Weight for masking loss (λ in paper, default: 1.0)
- `masked_loss_noise`: Noise level for masking loss (default: 1.0)
- `hidden_sizes`: MLP architecture (default: [128, 256, 128])

## Testing and Evaluation

### Evaluate Trained Models

The unified `eval.py` script supports both simulation and real robot evaluation:

```bash
# Evaluate on simulation (default)
python src/scripts/eval.py \
    --seed 12345 \
    --config config/reward_learning/obj20_sg10_persg5/maskedrl_llm_mask.yaml \
    -hc config/humans/frankarobot_multiple_humans.yaml \
    -dq 10

# Evaluate on real robot
python src/scripts/eval.py \
    --realrobot \
    --seed 12345 \
    --config config/reward_learning/realrobot/maskedrl_llm_mask.yaml \
    -hc config/humans/frankarobot_multiple_humans_validfeat1and2.yaml \
    -dq 10
```

**Using bash script wrapper**:

```bash
# Simulation
bash src/scripts/eval.sh \
    -t obj20_sg10_persg5 \
    -m maskedrl_llm_mask \
    -s 12345 \
    -d 10

# Real robot
bash src/scripts/eval.sh \
    -r \
    -t realrobot \
    -m maskedrl_llm_mask \
    -s 12345 \
    -d 10
```

### Visualization

Visualize trajectories:

```bash
python src/scripts/visualize_trajs.py \
    --traj-path data/traj_sets/frankarobot_obj20_sg10_persg5.npy
```

## Simulation vs Real Robot

### Simulation Setup

- **Environment**: PyBullet simulation of Franka Research 3 robot
- **State space**: 19 dimensions (3D end-effector position, 9D rotation matrix, 7D object positions)
- **Dataset**: `frankarobot_obj20_sg10_persg5.npy`
- **Config**: Use `frankarobot_multiple_humans.yaml` for human preferences
- **Training script**: `train.py` (unified script, default is simulation)

### Real Robot Setup

- **Hardware**: 7DoF Franka Research 3 robot arm
- **State space**: Same 19 dimensions as simulation
- **Dataset**: `frankarobot_obj100_sg50_persg50.npy`
- **Config**: Use `frankarobot_multiple_humans_validfeat1and2.yaml` for human preferences
- **Training script**: `train.py --realrobot` (unified script with real robot flag)

## Project Structure

```
Masked-IRL/
├── config/
│   ├── envs/              # Environment configurations
│   ├── humans/            # Human preference configurations
│   ├── reward_learning/   # Reward learning model configs
│   └── trajset_gen/       # Trajectory generation configs
├── data/
│   ├── traj_sets/         # Trajectory datasets
│   └── resources/         # Environment resources
├── src/
│   ├── envs/              # Environment implementations
│   ├── models/            # Model implementations
│   │   ├── humans/        # Human preference models
│   │   └── reward_learning/  # Reward learning models
│   ├── scripts/           # Training and evaluation scripts
│   └── utils/            # Utility functions
├── scripts/               # Additional utility scripts
└── requirements.txt       # Python dependencies
```

## Train and Test

- `src/scripts/train.py` - Unified training script (simulation by default, use `--realrobot` for real robot)
- `src/scripts/train.sh` - Bash wrapper for training with batch processing support
- `src/scripts/eval.py` - Unified evaluation script (simulation by default, use `--realrobot` for real robot)
- `src/scripts/eval.sh` - Bash wrapper for evaluation
- `src/scripts/train_and_eval.py` - Combined training and evaluation script with advanced options
- `src/scripts/run_local.sh` - Local training script with batch processing
- `src/scripts/generate_traj_sets.py` - Generate trajectory datasets
- `src/scripts/generate_trajs_obj_multiple_locations.py` - Generate trajectories for multiple object locations
- `src/scripts/visualize_trajs.py` - Visualize trajectories
- `src/scripts/slurm.sbatch` - SLURM job submission script

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hwang2026masked,
  title={Masked IRL: LLM-Guided Reward Disambiguation from Demonstrations and Language},
  author={Hwang, Minyoung and Forsey-Smerek, Alexandra and Dennler, Nathaniel and Bobu, Andreea},
  journal={{IEEE} International Conference on Robotics and Automation, {ICRA}},
  year={2026}
}
```

## License

See LICENSE file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository.
