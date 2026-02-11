"""
Visualize Trajectories with Learned Rewards

This script loads trained IRL models and visualizes trajectories ranked by learned rewards.
Supports filtering by features and replaying top/bottom trajectories.
"""

import os
import sys
import json
import time
import argparse
import yaml
import glob

sys.path.append("../")

import numpy as np
import torch

from src.utils.parser import *
from src.utils.bullet_utils import *
from src.utils.viz_utils import *
from src.utils.feature_utils import theta_to_language

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Model loading utilities
def parse_model_info(ckpt_path: str):
    """Parse model metadata from checkpoint path."""
    parent = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    irl_type = parent.split("_th34")[0]
    seed_dir = os.path.basename(os.path.dirname(ckpt_path))
    seed = int(seed_dir.replace("seed", ""))
    return {
        'irl_type': irl_type,
        'state_dim': 19,
        'hidden_sizes': [128, 256, 128],
        'seed': seed,
        'run_dir': os.path.dirname(os.path.dirname(ckpt_path))
    }

def discover_models(pattern):
    """Discover model checkpoints matching the pattern."""
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        print('No checkpoints found for pattern:', pattern)
    return ckpts

def build_irl_skeleton(info, llm_state_mask_path=None):
    """Build a minimal IRL model skeleton for loading weights."""
    irl_cfg = {
        'type': info['irl_type'],
        'hidden_sizes': info['hidden_sizes'],
        'lr': 1e-4,
        'batch_size': 256,
        'num_iterations': 0,
        'masked_loss_weight': 0.0,
        'masked_loss_noise': 0.0,
        'language_encoder': 't5',
        'language': {'vocab_size': 10000, 'batch_size': 512}
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_tensor = torch.zeros((1, 21, info['state_dim']), dtype=torch.float32)
    dummy_thetas = [[1,0,0,0,0]]
    irl = make_irl(irl_cfg, None, [], [],
                   train_states=dummy_tensor, demo_states=dummy_tensor,
                   demo_thetas=dummy_thetas, demo_indices=None,
                   train_thetas=dummy_thetas, wandb=False,
                   human_win_rates=None, seen_theta_human_win_rates=None,
                   test_trajs=dummy_tensor, save_path=None,
                   test_features=dummy_tensor, test_states=dummy_tensor,
                   language_ambiguity=None, unseen_humans=None,
                   finetune_demo_features=dummy_tensor, finetune_demo_states=dummy_tensor,
                   finetune_demo_thetas=dummy_thetas,
                   llm_state_mask_path=llm_state_mask_path)
    return irl

def load_irl_from_ckpt(ckpt_path, info, llm_state_mask_path=None):
    """Load IRL model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(ckpt_path, map_location=device)
    if hasattr(data, 'state_dict') or hasattr(data, 'forward'):
        irl = data
    else:
        irl = build_irl_skeleton(info, llm_state_mask_path)
        if isinstance(data, dict):
            if 'cost_nn' in data and hasattr(irl, 'cost_nn'):
                irl.cost_nn.load_state_dict(data['cost_nn'])
            if 'lang_encoder' in data and hasattr(irl, 'lang_encoder'):
                irl.lang_encoder.load_state_dict(data['lang_encoder'])
    if hasattr(irl, 'to'):
        irl = irl.to(device)
    return irl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize trajectories ranked by learned rewards from trained IRL models."
    )
    parser.add_argument("--traj-path", default="../data/traj_sets/frankarobot_obj20_sg10_persg5.npy",
                       help="Path to trajectory dataset")
    parser.add_argument("--model-glob", default="../data/models/irl/*/seed*/irl_last.pt",
                       help="Glob pattern to find model checkpoints")
    parser.add_argument("--env-config", default="../config/envs/frankarobot.yaml",
                       help="Environment configuration file")
    parser.add_argument("--theta", type=int, nargs=5, default=[-1, 0, 1, 0, 0],
                       help="Theta vector [table, human, laptop, proxemics, coffee]")
    parser.add_argument("--model-index", type=int, default=0,
                       help="Index of model checkpoint to use")
    parser.add_argument("--n", type=int, default=3,
                       help="Number of top trajectories to visualize")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--show-ee-coords", action="store_true", default=False,
                       help="Show end-effector coordinates")
    parser.add_argument("--debug", action="store_true", help="Drop into ipdb at end")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load trajectories
    print(f"Loading trajectories from {args.traj_path}")
    traj_set = np.load(args.traj_path, allow_pickle=True)
    print(f"Trajectory set shape: {traj_set.shape}")

    # Load environment config
    with open(args.env_config, "r") as stream:
        params = yaml.safe_load(stream)

    env = make_env(params["env"])
    all_trajs, train_trajs, test_trajs = load_split_data(
        args.traj_path,
        params["env"]["per_SG"],
        params["env"]["train_test_split"]
    )
    print(f"Total: {len(all_trajs)}, Train: {len(train_trajs)}, Test: {len(test_trajs)}")

    # Setup human for feature/state calculation
    human_params = {
        "features": ["table", "human", "laptop", "proxemics", "coffee"],
        "feature_scaling": "normalize",
        "preferencer": {
            "theta": [1, 0, 0, 0, 0],
            "beta": 20.0,
            "f_method": "boltzmann",
            "s_method": "luce"
        },
        "type": params["env"]["type"]
    }
    human = make_human(human_params, env, all_trajs)
    all_states = np.array([human.calc_eef_object_states(traj, state_dim=19) for traj in all_trajs])

    # Discover and load model
    ckpts = discover_models(args.model_glob)
    print(f'Found {len(ckpts)} checkpoints')
    
    if len(ckpts) == 0:
        print("No checkpoints found. Exiting.")
        sys.exit(1)
    
    if args.model_index >= len(ckpts):
        print(f"Model index {args.model_index} out of range. Using index 0.")
        args.model_index = 0
    
    ckpt = ckpts[args.model_index]
    theta = args.theta
    print(f"Using theta: {theta}")
    print(f"Loading model from: {ckpt}")

    # Load model
    info = parse_model_info(ckpt)
    model_name = f"{info['irl_type']}_seed{info['seed']}"
    print(f'\nModel: {model_name}')
    print(f'  Type: {info["irl_type"]} | State dim: {info["state_dim"]} | Hidden: {info["hidden_sizes"]} | Seed: {info["seed"]}')
    
    irl = load_irl_from_ckpt(ckpt, info)
    
    # Compute learned rewards for all trajectories
    N = len(all_states)
    inst = theta_to_language([theta])[0]
    emb = irl.lang_encoder(inst).to(torch.float64)
    emb_batch = emb.expand(N, -1)
    all_states_tensor = torch.as_tensor(all_states, dtype=torch.float32, device=device)
    learned_rewards = -irl.calc_traj_cost_batch(all_states_tensor, emb_batch).detach().cpu().numpy()
    
    # Normalize rewards for visualization
    learned_rewards = (learned_rewards - np.min(learned_rewards)) / (np.max(learned_rewards) - np.min(learned_rewards) + 1e-8)

    # Select top N trajectories (highest rewards)
    top_n_idx = np.argsort(learned_rewards)[-args.n:]
    top_n_trajs = all_trajs[top_n_idx]
    
    print(f"\nLanguage instruction: {inst}")
    print(f"Top {args.n} trajectory indices: {top_n_idx}")
    print(f"Top {args.n} rewards: {learned_rewards[top_n_idx]}")

    # Set camera view
    xview(view=0)
    time.sleep(0.1)

    # Replay trajectories
    show_ee_coords = args.show_ee_coords
    env.replay(
        top_n_trajs,
        colors=[[0, 0, 1] for _ in range(len(top_n_trajs))],
        show_ee_coords=show_ee_coords
    )

    if args.debug:
        import ipdb; ipdb.set_trace()

    p.removeAllUserParameters()
    p.disconnect()