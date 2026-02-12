import os
import sys
import random
import yaml
import numpy as np
import torch
from scipy import special
import argparse
import copy
import json
from tqdm import tqdm

sys.path.insert(1, '../')

from src.utils.parser import *

def calculate_win_rate(test_trajs, ground_truth_reward, learned_reward):
    """
    Calculate the win rate by comparing preference labels from ground truth and learned rewards.

    Args:
        test_trajs: List of trajectory pairs. Each entry is a tuple (traj1, traj2).
        ground_truth_reward: Function that calculates the ground truth reward for a trajectory.
        learned_reward: Function that calculates the learned reward for a trajectory.

    Returns:
        Win rate as a float.
    """
    win_count = 0
    # sample traj pairs from the set of test trajectories: choose M pairs of trajectories
    num_samples = 100
    traj_pairs = [(test_trajs[i], test_trajs[j]) for i in range(len(test_trajs)) for j in range(i+1, len(test_trajs))]
    traj_pairs = random.sample(traj_pairs, num_samples)
    total_count = len(traj_pairs)

    for traj1, traj2 in traj_pairs:
        # Compute ground truth rewards
        gt_reward_1 = ground_truth_reward(traj1)
        gt_reward_2 = ground_truth_reward(traj2)
        gt_preference = gt_reward_1 > gt_reward_2

        # Compute learned rewards
        lr_reward_1 = learned_reward(traj1)
        lr_reward_2 = learned_reward(traj2)
        lr_preference = lr_reward_1 > lr_reward_2

        # Check if preferences match
        if gt_preference == lr_preference:
            win_count += 1

    return win_count / total_count

def evaluate_irl_with_test_trajs(test_trajs, ground_truth_reward, learned_reward):
    """
    Evaluate the performance of the learned reward function using provided test trajectories.

    Args:
        test_trajs: List of trajectory pairs for evaluation.
        ground_truth_reward: Function to compute ground truth rewards.
        learned_reward: Function to compute learned rewards.

    Prints:
        The win rate of the learned reward model.
    """
    print("Calculating win rate...")
    win_rate = calculate_win_rate(test_trajs, ground_truth_reward, learned_reward)

    print(f"Win rate: {win_rate * 100:.2f}%")
    return win_rate

def load_split_data(trajset_file, per_SG, train_test_split=0.8, indices_file=None, realrobot=False):
    """
    Load trajectories and split into training and test sets using consistent indices.
    """
    all_trajs = np.load(trajset_file)
    orig_shape = all_trajs[0].shape

    if not realrobot:
        # Reshape to account for per_SG grouping (simulation only)
        all_trajs = all_trajs.reshape((-1, int(per_SG), *orig_shape))
        all_trajs = all_trajs.reshape((-1, *orig_shape))
        assert len(all_trajs) % per_SG == 0, "The total trajectory count must be divisible by per_SG."

    if indices_file is not None and os.path.exists(indices_file):
        print(f"Loading saved split indices from {indices_file}")
        with open(indices_file, 'r') as f:
            split_indices = json.load(f)
        train_indices = split_indices["train_indices"]
        test_indices = split_indices["test_indices"]
    else:
        if realrobot:
            raise NotImplementedError("Please provide a valid indices_file with existing split indices for real robot.")
        # Auto-split for simulation
        indices = np.arange(len(all_trajs))
        np.random.shuffle(indices)
        train_size = int(train_test_split * len(all_trajs))
        train_size = (train_size // per_SG) * per_SG
        train_indices = indices[:train_size].tolist()
        test_indices = indices[train_size:].tolist()
        if indices_file is not None:
            with open(indices_file, 'w') as f:
                json.dump({"train_indices": train_indices, "test_indices": test_indices}, f)
            print(f"Saved new split indices to {indices_file}")
    
    train_trajs = all_trajs[train_indices]
    test_trajs = all_trajs[test_indices]
    
    return all_trajs, train_trajs, test_trajs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate IRL with demos from multiple humans')
    parser.add_argument('-c', '--config', type=str, required=True, help='config file')
    parser.add_argument('-hc', '--human_config', type=str, default=None, help='human config file')
    parser.add_argument('-fd', '--featurizer_dir', type=str, default="../data/models", help='save folder')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-pre', '--pretrain_amt', type=int, default=0, help='pretrain data amount')
    parser.add_argument('-sq', '--sim_queries', type=int, default=100, help='similarity query amount')
    parser.add_argument('-dq', '--demo_queries', type=int, default=10, help='demo query amount')
    parser.add_argument('--realrobot', action='store_true', help='Use real robot mode')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        params = yaml.safe_load(stream)

    # Set defaults based on realrobot flag
    if args.realrobot:
        if args.human_config is None:
            args.human_config = "../config/humans/frankarobot_multiple_humans_validfeat1and2.yaml"
    else:
        if args.human_config is None:
            args.human_config = "../config/humans/frankarobot_multiple_humans.yaml"

    set_seed(args.seed)
    env = make_env(params["env"])

    # Load data and split into train and test
    indices_file = os.path.join(params["irl"]["data_split_config_path"], "split_indices.json") if "irl" in params else None
    all_trajs, train_trajs, test_trajs = load_split_data(
        params["env"]["trajset_file"], 
        params["env"]["per_SG"], 
        params["env"]["train_test_split"],
        indices_file=indices_file,
        realrobot=args.realrobot
    )
    print(len(all_trajs), len(test_trajs), len(train_trajs))

    # Load human configs
    with open(args.human_config, "r") as stream:
        humans_params_list = yaml.safe_load(stream)
    
    results = []
    for human_params in tqdm(humans_params_list["humans"]):
        human_params["type"] = params["env"]["type"]
        print(human_params)
        human = make_human(human_params, env, train_trajs)

        # Generate human data
        demos, demo_indices = human.generate_demos(args.demo_queries)
        train_features = np.array([human.calc_features(traj) for traj in train_trajs])
        demo_features = np.array([human.calc_features(traj) for traj in demos])
        featurizer = None

        # mask
        human_theta = human.theta
        state_mask = [1 if abs(human_theta[i]) > 0.1 else 0 for i in range(len(human_theta))]
        params["irl"]["state_mask"] = state_mask

        irl = make_irl(params["irl"], featurizer, demos, train_trajs, train_features=train_features, demo_features=demo_features)
        irl.cost_nn.state_dict()['net.0.weight'][0] = 0.0  # custom initialization

        irl.train(params["irl"]["num_iterations"], save_dir="../data/models/irl")

        print("Avg Demo Cost: ", np.mean(irl.calc_cost(demos, demo_features)))
        print("Avg Traj Cost: ", np.mean(irl.calc_cost(train_trajs, train_features)))

        # Evaluate learned reward
        def ground_truth_reward(traj):
            return -np.dot(human.theta, human.calc_features(traj))
        
        def learned_reward(traj):
            return -irl.calc_cost([traj], [human.calc_features(traj)])[0]
        
        win_rate = evaluate_irl_with_test_trajs(test_trajs, ground_truth_reward, learned_reward)
        results.append(win_rate)

    # Save results
    results_path = "../data/results/frankarobot_results_0319_{}_full_features_demos{}.json".format(params["irl"]["type"], args.demo_queries)
    if args.realrobot:
        results_path = results_path.replace("frankarobot_results", "frankarobot_realrobot_results")
    
    results_list = copy.deepcopy(humans_params_list["humans"])
    for i, human_params in enumerate(results_list):
        results_list[i]["win_rate"] = results[i]
        results_list[i]["seed"] = args.seed

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_list, f, indent=4)
    
    print(f"Results saved to {results_path}")
