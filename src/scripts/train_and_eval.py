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
import hashlib
from tqdm import tqdm
import joblib
import wandb

sys.path.insert(1, '../')

from src.utils.parser import *
from src.utils.feature_utils import theta_to_language
from src.utils.eval_utils import calculate_win_rate

def jsonNpEncoder(object):
    if isinstance(object, np.generic):
        return object.item()
    elif isinstance(object, np.ndarray):
        return object.tolist()
    raise TypeError(f"Object of type {object.__class__.__name__} is not JSON serializable")

def load_split_data(trajset_file, per_SG, train_test_split=0.8, indices_file=None, multiple_objs=False):
    """
    Load trajectories and split into training and test sets using consistent indices.
    If multiple_objs is True, loads from dictionary format instead of array.
    """
    if multiple_objs:
        all_data = np.load(trajset_file, allow_pickle=True).item()
        return all_data
    else:
        all_trajs = np.load(trajset_file)
        orig_shape = all_trajs[0].shape

        # Reshape to account for per_SG grouping
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

def get_theta_key(theta):
    """Generate an interpretable key for a given theta vector."""
    return '_'.join(str(x) for x in theta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate IRL with aggregated demos from multiple humans')
    parser.add_argument('-c', '--config', type=str, required=True, help='config file')
    parser.add_argument('-hc', '--human_config', type=str, default="../config/humans/frankarobot_multiple_humans_simple.yaml",
                        help='human config file')
    parser.add_argument('-fd', '--featurizer_dir', type=str, default="../data/models", help='save folder')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-pre', '--pretrain_amt', type=int, default=0, help='pretrain data amount')
    parser.add_argument('-sq', '--sim_queries', type=int, default=100, help='similarity query amount')
    parser.add_argument('-dq', '--demo_queries', type=int, default=10, help='demo query amount')
    parser.add_argument('--demo_indices_file', type=str, default=None, help='File for saving/loading all demo indices by theta')
    parser.add_argument('--preprocessed_traj_data', type=str, default=None, help='File for saving/loading preprocessed trajectory data')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    
    # Split mode arguments
    parser.add_argument('--split_mode', type=str, default='none', choices=['none', 'theta_keys', 'human_indices'],
                        help='How to split humans: none (all), theta_keys (by theta keys), human_indices (by human index)')
    parser.add_argument('--num_train_thetas', type=int, default=None, help='Number of training thetas (for theta_keys mode)')
    parser.add_argument('--human_indices_split_path', type=str, default=None,
                        help='Path to JSON file with human/theta split information')
    
    # Additional options
    parser.add_argument('--film', type=str, default="0", help='Film layer configuration (e.g., "0" or "0_1")')
    parser.add_argument('--multiple_objs', action='store_true', help='Use dictionary format for trajectories (multiple objects mode)')
    parser.add_argument('--omit_referent', action='store_true', help='Omit referent in language (for ambiguity experiments)')
    parser.add_argument('--omit_expression', action='store_true', help='Omit expression in language (for ambiguity experiments)')
    
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        params = yaml.safe_load(stream)

    # Set omit flags in params if provided
    if args.omit_referent:
        params["irl"]["omit_referent"] = True
    if args.omit_expression:
        params["irl"]["omit_expression"] = True

    set_seed(args.seed)
    env = make_env(params["env"])

    # Set default demo_indices_file if not provided
    if args.demo_indices_file is None:
        if args.multiple_objs:
            args.demo_indices_file = os.path.join(params["irl"]["data_split_config_path"], f"demo_indices.json")
        else:
            args.demo_indices_file = os.path.join(params["irl"]["data_split_config_path"], f"demo_indices_100_{args.seed}.json")

    if args.preprocessed_traj_data is None:
        args.preprocessed_traj_data = os.path.join(params["irl"]["data_split_config_path"], f"preprocessed_traj_data_{args.seed}.pkl")
    
    scaling_coeffs_path = os.path.join(params["irl"]["data_split_config_path"], f"scaling_coeffs_{args.seed}.pkl")
    if os.path.exists(scaling_coeffs_path):
        scaling_coeffs = joblib.load(scaling_coeffs_path)
        print(f"Loaded scaling coefficients from {scaling_coeffs_path}")
    else:
        scaling_coeffs = None

    # Load trajectories
    if args.multiple_objs:
        all_data = load_split_data(params["env"]["trajset_file"], params["env"]["per_SG"],
                                   params["env"]["train_test_split"], None, multiple_objs=True)
        all_scaled_feats = all_data['all_scaled_feats']
        all_scaled_states = all_data['all_scaled_states']
        seen_objs_train_trajs_indices = all_data['seen_objs_train_trajs_indices']
        seen_objs_test_trajs_indices = all_data['seen_objs_test_trajs_indices']
        unseen_objs_test_trajs_indices = all_data['unseen_objs_test_trajs_indices']
        train_feats = all_scaled_feats[seen_objs_train_trajs_indices]
        train_states = all_scaled_states[seen_objs_train_trajs_indices]
        test_feats = all_scaled_feats[seen_objs_test_trajs_indices]
        test_states = all_scaled_states[seen_objs_test_trajs_indices]
        unseen_objs_test_feats = all_scaled_feats[unseen_objs_test_trajs_indices]
        unseen_objs_test_states = all_scaled_states[unseen_objs_test_trajs_indices]
        train_trajs = None  # Not used in multiple_objs mode
        test_trajs = None
    else:
        if not os.path.exists(params["irl"]["data_split_config_path"]):
            os.makedirs(params["irl"]["data_split_config_path"])
        indices_file = os.path.join(params["irl"]["data_split_config_path"], "split_indices.json")
        all_trajs, train_trajs, test_trajs = load_split_data(params["env"]["trajset_file"],
                                                              params["env"]["per_SG"],
                                                              params["env"]["train_test_split"],
                                                              indices_file=indices_file,
                                                              multiple_objs=False)
        print("Total trajectories:", len(all_trajs), "Train trajectories:", len(train_trajs), "Test trajectories:", len(test_trajs))
    
    traj_info = params["env"]["trajset_file"].split("/")[-1].split(".")[0]

    # Load demo indices
    if os.path.exists(args.demo_indices_file):
        with open(args.demo_indices_file, "r") as f:
            saved_demo_indices = json.load(f)
        for key in saved_demo_indices.keys():
            saved_demo_indices[key] = saved_demo_indices[key][:args.demo_queries]
        print(f"Loaded demo indices from {args.demo_indices_file}")
    else:
        saved_demo_indices = {}

    if os.path.exists(args.preprocessed_traj_data):
        print(f"Loading preprocessed trajectory data from {args.preprocessed_traj_data}")
        preprocessed_traj_data = joblib.load(args.preprocessed_traj_data)
    else:
        preprocessed_traj_data = {}
    
    # Load human configs
    with open(args.human_config, "r") as stream:
        humans_params_list = yaml.safe_load(stream)

    # Initialize aggregation lists
    agg_demos = []
    agg_demo_features = []
    agg_demo_indices = []
    agg_demo_states = []
    agg_demo_thetas = []
    agg_train_trajs = []
    agg_train_features = []
    agg_train_states = []
    agg_train_thetas = []
    human_win_rates = []
    seen_theta_human_win_rates = []
    
    # Load split information if needed
    if args.split_mode == 'theta_keys':
        if args.human_indices_split_path is None:
            args.human_indices_split_path = "../config/humans/thetas_sampled_data.json"
        human_indices_split = json.load(open(args.human_indices_split_path, "r"))
        if "sampled_thetas" in human_indices_split:
            human_indices_split_train = human_indices_split["sampled_thetas"]
            if args.num_train_thetas is not None:
                human_indices_split_train = human_indices_split_train[:args.num_train_thetas]
        else:
            human_indices_split_train = human_indices_split["train_thetas"]
            if args.num_train_thetas is not None:
                human_indices_split_train = human_indices_split_train[:args.num_train_thetas]
        human_indices_split_test = human_indices_split.get("test_thetas_30", human_indices_split.get("test_thetas", []))
        human_indices_split_train_theta_keys = [get_theta_key(theta) for theta in human_indices_split_train]
        human_indices_split_test_theta_keys = [get_theta_key(theta) for theta in human_indices_split_test]
    elif args.split_mode == 'human_indices':
        if args.human_indices_split_path is None:
            args.human_indices_split_path = "../config/humans/human_indices_split.json"
        human_indices_split = json.load(open(args.human_indices_split_path, "r"))
    
    print("Make humans to get demo indices...")
    
    for human_idx, human_params in tqdm(enumerate(humans_params_list["humans"])):
        human_params["type"] = params["env"]["type"]
        theta = human_params["preferencer"]["theta"]
        theta_key = get_theta_key(theta)

        # Determine if this human should be in train or test split
        in_train = False
        in_test = False
        
        if args.split_mode == 'none':
            in_train = True
        elif args.split_mode == 'theta_keys':
            if theta_key in human_indices_split_train_theta_keys:
                in_train = True
            elif theta_key in human_indices_split_test_theta_keys:
                in_test = True
        elif args.split_mode == 'human_indices':
            if human_idx in human_indices_split["train"]:
                in_train = True
            elif human_idx in human_indices_split["test"]:
                in_test = True
            else:
                if args.multiple_objs:
                    raise ValueError(f"Human index {human_idx} not in train or test split.")
                else:
                    print(f"Human index {human_idx} not in train or test split. Skipping...")
                    continue

        if args.multiple_objs:
            # Multiple objects mode - use preprocessed features/states
            demo_indices = saved_demo_indices[theta_key]
            print(f"Loaded saved demo indices for theta {theta} with key {theta_key}")
            
            if in_train:
                demo_feats = all_scaled_feats[demo_indices]
                demo_states = all_scaled_states[demo_indices]
                human_theta = theta
                demo_thetas = [human_theta for _ in range(len(demo_indices))]
                train_thetas = [human_theta for _ in range(len(train_feats))]
                
                agg_demo_features.extend(demo_feats)
                agg_demo_states.extend(demo_states)
                agg_demo_thetas.extend(demo_thetas)
                agg_train_features.extend(train_feats)
                agg_train_states.extend(train_states)
                agg_train_thetas.extend(train_thetas)
                
                seen_theta_human_win_rates.append({"theta": human_theta})
            elif in_test:
                human_theta = theta
                human_win_rates.append({"theta": human_theta})
        else:
            # Standard mode - compute features/states
            if "featurized_trajs" not in preprocessed_traj_data:
                human = make_human(human_params, env, train_trajs)
                preprocessed_traj_data = {
                    "featurized_trajs": human.featurized_trajs,
                    "probs": human.probs,
                }
            else:
                human = make_human(human_params, env, train_trajs, **preprocessed_traj_data, scaling_coeffs=scaling_coeffs)

            if not os.path.exists(scaling_coeffs_path):
                joblib.dump(human.scaling_coeffs, scaling_coeffs_path)
                print(f"Saved scaling coefficients to {scaling_coeffs_path}")
            
            if theta_key in saved_demo_indices:
                demo_indices = saved_demo_indices[theta_key]
                demos = [train_trajs[i] for i in demo_indices]
                print(f"Loaded saved demo indices for theta {theta} with key {theta_key}")
            else:
                if not hasattr(human, 'preferencer'):
                    human.set_trajset(train_trajs)
                    human.set_preference(human_params["preferencer"])
                demos, demo_indices = human.generate_demos(args.demo_queries)
                saved_demo_indices[theta_key] = demo_indices
                print(f"Generated and saved demo indices for theta {theta} with key {theta_key}")

            if in_train:
                demo_feats = human.featurized_trajs[demo_indices]
                demo_states = [human.calc_eef_object_states(traj) for traj in demos]
                train_feats = human.featurized_trajs
                train_states = [human.calc_eef_object_states(traj) for traj in train_trajs]
                human_theta = theta
                demo_thetas = [human_theta for _ in range(len(demos))]
                train_thetas = [human_theta for _ in range(len(train_trajs))]
                
                agg_demos.extend(demos)
                agg_demo_features.extend(demo_feats)
                agg_demo_states.extend(demo_states)
                agg_demo_thetas.extend(demo_thetas)
                agg_demo_indices.extend(demo_indices)
                agg_train_trajs.extend(train_trajs)
                agg_train_features.extend(train_feats)
                agg_train_states.extend(train_states)
                agg_train_thetas.extend(train_thetas)
                
                seen_theta_human_win_rates.append({"theta": human_theta, "demos": demos})
            elif in_test:
                human_theta = theta
                human_win_rates.append({"theta": human_theta, "demos": demos})

            if "test_features" not in preprocessed_traj_data and test_trajs is not None:
                test_features = [human.calc_features(traj) for traj in test_trajs]
                test_states = [human.calc_eef_object_states(traj) for traj in test_trajs]
                test_features = np.array(test_features)
                test_states = np.array(test_states)
                preprocessed_traj_data["test_features"] = test_features
                preprocessed_traj_data["test_states"] = test_states
                joblib.dump(preprocessed_traj_data, args.preprocessed_traj_data)
                print(f"Saved preprocessed trajectory data to {args.preprocessed_traj_data}")
    
    # Save demo indices
    if not os.path.exists(args.demo_indices_file):
        with open(args.demo_indices_file, "w") as f:
            json.dump(saved_demo_indices, f, indent=4, default=jsonNpEncoder)
        print(f"Saved all demo indices to {args.demo_indices_file}")

    # Save preprocessed data
    if not os.path.exists(args.preprocessed_traj_data) and not args.multiple_objs:
        with open(args.preprocessed_traj_data, "wb") as f:
            joblib.dump(preprocessed_traj_data, f)
        print(f"Saved preprocessed trajectory data to {args.preprocessed_traj_data}")
    
    train_features = np.array(agg_train_features) if agg_train_features else None
    train_states = np.array(agg_train_states)
    
    # wandb setup
    dataset = params["env"]["trajset_file"].split("/")[-1].split(".")[0]
    state_dim = train_states.shape[-1] if len(train_states.shape) > 1 else train_states.shape[0]
    if args.wandb:
        wandb.login(key=[line.strip() for line in open(os.path.expanduser("~/.wandb_api_key"))][0])
        if 'masked' in params['irl']['type']:
            wandb_run_name = f"{dataset}_sdim{state_dim}_{params['irl']['type']}_film{args.film}_demos{args.demo_queries}_it{params['irl']['num_iterations']}_lr{params['irl']['lr']}_hidden{'_'.join(map(str, params['irl']['hidden_sizes']))}_mweight{params['irl']['masked_loss_weight']}_mnoise{params['irl']['masked_loss_noise']}_seed{args.seed}"
        else:
            wandb_run_name = f"{dataset}_sdim{state_dim}_{params['irl']['type']}_film{args.film}_demos{args.demo_queries}_it{params['irl']['num_iterations']}_lr{params['irl']['lr']}_hidden{'_'.join(map(str, params['irl']['hidden_sizes']))}_seed{args.seed}"
        
        if args.omit_referent:
            wandb_run_name = wandb_run_name.replace("seed{args.seed}", f"omit_referent_seed{args.seed}")
        elif args.omit_expression:
            wandb_run_name = wandb_run_name.replace("seed{args.seed}", f"omit_expression_seed{args.seed}")
        
        project_name = 'Masked-IRL-5' if args.split_mode != 'none' else 'Masked-IRL'
        wandb.init(project=project_name, config=params, name=wandb_run_name, reinit=True)
        wandb.config.update({
            "seed": args.seed,
            "demo_queries": args.demo_queries,
            "config": params,
            "args": vars(args),
            "state_dim": state_dim,
        })
    
    # Model path
    hidden_sizes = params["irl"]["hidden_sizes"]
    hidden_sizes_str = "_".join(map(str, hidden_sizes))
    if 'masked' in params["irl"]["type"]:
        model_path = "../data/models/irl/{}/{}_demos{}_it{}_lr{}_hidden{}_mweight{}_mnoise{}_seed{}".format(
            traj_info, params["irl"]["type"], args.demo_queries, params["irl"]["num_iterations"],
            params["irl"]["lr"], hidden_sizes_str, params["irl"]["masked_loss_weight"],
            params["irl"]["masked_loss_noise"], args.seed)
    else:
        model_path = "../data/models/irl/{}/{}_demos{}_it{}_lr{}_hidden{}_seed{}".format(
            traj_info, params["irl"]["type"], args.demo_queries, params["irl"]["num_iterations"],
            params["irl"]["lr"], hidden_sizes_str, args.seed)
    if args.debug:
        model_path = model_path + "_debug"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create IRL model
    film_config = [int(x) for x in args.film.split("_")] if args.film else None
    
    if args.multiple_objs:
        irl = make_irl(params["irl"], None, None, None,
                      train_states=train_states,
                      demo_states=np.array(agg_demo_states),
                      demo_thetas=np.array(agg_demo_thetas),
                      train_thetas=np.array(agg_train_thetas),
                      wandb=args.wandb,
                      human_win_rates=human_win_rates,
                      seen_theta_human_win_rates=seen_theta_human_win_rates,
                      test_trajs=None,
                      save_path=model_path,
                      test_features=test_feats,
                      test_states=test_states,
                      )
    else:
        irl = make_irl(params["irl"], None, agg_demos, agg_train_trajs,
                      train_states=train_states,
                      demo_states=np.array(agg_demo_states),
                      demo_thetas=np.array(agg_demo_thetas),
                      demo_indices=np.array(agg_demo_indices) if agg_demo_indices else None,
                      train_thetas=np.array(agg_train_thetas),
                      wandb=args.wandb,
                      human_win_rates=human_win_rates,
                      seen_theta_human_win_rates=seen_theta_human_win_rates if seen_theta_human_win_rates else None,
                      test_trajs=test_trajs,
                      save_path=model_path,
                      test_features=preprocessed_traj_data.get("test_features"),
                      test_states=preprocessed_traj_data.get("test_states"),
                      omit_referent=params["irl"].get("omit_referent", False),
                      omit_expression=params["irl"].get("omit_expression", False),
                      )
    
    # Train
    irl.train(params["irl"]["num_iterations"])

    # Evaluate if not in multiple_objs mode
    if not args.multiple_objs and args.split_mode == 'none':
        if 'masked' in params["irl"]["type"]:
            results_path = "../data/results/{}/frankarobot_results_0415_{}_conditioned_on_language_states_input_demos{}_hidden{}_mweight{}_mnoise{}_seed{}.json".format(
                traj_info, params["irl"]["type"], args.demo_queries, hidden_sizes_str,
                params["irl"]["masked_loss_weight"], params["irl"]["masked_loss_noise"], args.seed)
        else:
            results_path = "../data/results/{}/frankarobot_results_0415_{}_conditioned_on_language_states_input_demos{}_hidden{}_seed{}.json".format(
                traj_info, params["irl"]["type"], args.demo_queries, hidden_sizes_str, args.seed)
        
        if args.debug:
            results_path = results_path.replace(".json", "_debug.json")
        if not os.path.exists(os.path.dirname(results_path)):
            os.makedirs(os.path.dirname(results_path))
        results_list = copy.deepcopy(humans_params_list["humans"])
        results, avg_win_rate = irl.evaluate()
        for i, human_params in enumerate(results_list):
            human_params["win_rate"] = results[i]
        with open(results_path, "w") as f:
            json.dump(results_list, f, indent=4, default=jsonNpEncoder)
        print(f"Results saved to {results_path}")
    
    if args.wandb:
        wandb.finish()
        print("WANDB FINISHED")
