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
from src.utils.feature_utils import theta_to_state_mask, theta_to_llm_state_mask

def jsonNpEncoder(object):
    if isinstance(object, np.generic):
        return object.item()
    elif isinstance(object, np.ndarray):
        return object.tolist()
    raise TypeError(f"Object of type {object.__class__.__name__} is not JSON serializable")

def load_split_data(trajset_file, per_SG, train_test_split=0.8, indices_file=None, realrobot=False):
    """
    Load trajectories and split into training and test sets using consistent indices.
    If the indices file exists, load the saved train and test indices.
    Otherwise, randomly shuffle and compute the indices, and save them (if indices_file is provided).
    
    Args:
        realrobot: If True, skip reshaping and require indices_file. If False, reshape trajectories.
    """
    all_trajs = np.load(trajset_file)
    orig_shape = all_trajs[0].shape

    if not realrobot:
        # Reshape to account for per_SG grouping (simulation only)
        all_trajs = all_trajs.reshape((-1, int(per_SG), *orig_shape))
        # Flatten back to (num_samples, shape)
        all_trajs = all_trajs.reshape((-1, *orig_shape))
        # Ensure the total number of trajectories is divisible by per_SG
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
        # Compute training size, adjusted to be a multiple of per_SG
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
    """
    Generate an interpretable key for a given theta vector (assumed to be 5-dimensional with elements in {-1, 0, 1}).
    For example, a theta of [-1, 0, 1, 1, 0] will yield the key "-1_0_1_1_0".
    """
    return '_'.join(str(x) for x in theta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train IRL with aggregated demos from multiple humans')
    parser.add_argument('-c', '--config', type=str, required=True, help='config file')
    parser.add_argument('-hc', '--human_config', type=str, default=None, help='human config file')
    parser.add_argument('-fd', '--featurizer_dir', type=str, default="../data/models", help='save folder')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-pre', '--pretrain_amt', type=int, default=0, help='pretrain data amount')
    parser.add_argument('-sq', '--sim_queries', type=int, default=100, help='similarity query amount')
    parser.add_argument('-dq', '--demo_queries', type=int, default=10, help='demo query amount')
    parser.add_argument('--demo_indices_file', type=str, default=None, help='File for saving/loading all demo indices by theta')
    parser.add_argument('--preprocessed_traj_data', type=str, default=None, help='File for saving/loading preprocessed trajectory data')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('-nt', '--num_train_thetas', type=int, default=100, help='number of training thetas')
    parser.add_argument('--test_trajs_set_path', type=str, default=None, help='Path to the test trajectories set')
    parser.add_argument('--test_seed', type=int, default=12345, help='seed for test trajs set')
    parser.add_argument('--state_dim', type=int, default=9, help='state dimension for EEF object states')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--num_iterations', type=int, default=None, help='number of training iterations')
    parser.add_argument('--llm_state_mask_path', type=str, default=None, help='llm state mask path')
    parser.add_argument('--language_ambiguity', type=str, default=None, help='language ambiguity: None, omit-referent, omit-expression, or paraphrase')
    parser.add_argument('--llm_disambiguation', type=str, default=None, help='llm disambiguation: None, llm, or vlm (simulation only)')
    parser.add_argument('--realrobot', action='store_true', help='Use real robot mode (different defaults and behavior)')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        params = yaml.safe_load(stream)

    if args.lr is not None:
        params["irl"]["lr"] = args.lr
    if args.batch_size is not None:
        params["irl"]["batch_size"] = args.batch_size
    if args.num_iterations is not None:
        params["irl"]["num_iterations"] = args.num_iterations

    # Set defaults based on realrobot flag
    if args.realrobot:
        if args.human_config is None:
            args.human_config = "../config/humans/frankarobot_multiple_humans_validfeat1and2.yaml"
        if args.demo_indices_file is None:
            args.demo_indices_file = "../config/data_split_config/frankarobot_real/demo_indices_from_all_traj.json"
        if args.test_trajs_set_path is None:
            args.test_trajs_set_path = "../config/data_split_config/frankarobot_real/test_paired_trajs.npy"
        human_indices_split_path = "../config/humans/thetas_sampled_data_realrobot.json"
        # Real robot doesn't support llm_disambiguation
        if args.llm_disambiguation is not None:
            print("Warning: llm_disambiguation is not supported for real robot. Ignoring.")
            args.llm_disambiguation = None
    else:
        if args.human_config is None:
            args.human_config = "../config/humans/frankarobot_multiple_humans_simple.yaml"
        if args.demo_indices_file is None:
            args.demo_indices_file = os.path.join(params["irl"]["data_split_config_path"], f"demo_indices_100_{args.seed}.json")
        if args.test_trajs_set_path is None:
            args.test_trajs_set_path = "../config/data_split_config/frankarobot_obj100_sg50_persg50/test_trajs_set.npy"
        human_indices_split_path = "../config/humans/thetas_sampled_data_valid_features_1_table_human_laptop.json"

    set_seed(args.seed)
    env = make_env(params["env"])

    scaling_coeffs_path = os.path.join(params["irl"]["data_split_config_path"], f"scaling_coeffs_{args.seed}.pkl")
    if os.path.exists(scaling_coeffs_path):
        scaling_coeffs = joblib.load(scaling_coeffs_path)
        print(f"Loaded scaling coefficients from {scaling_coeffs_path}")
    else:
        scaling_coeffs = None

    # Load trajectories with consistent train/test split
    if not os.path.exists(params["irl"]["data_split_config_path"]):
        os.makedirs(params["irl"]["data_split_config_path"])
    indices_file = os.path.join(params["irl"]["data_split_config_path"], "split_indices.json")
    
    if args.realrobot:
        # Real robot requires existing split indices
        train_test_split = json.load(open(indices_file, "r"))
    
    all_trajs, train_trajs, dummy_test_trajs = load_split_data(params["env"]["trajset_file"],
                                                          params["env"]["per_SG"],
                                                          params["env"]["train_test_split"],
                                                          indices_file=indices_file,
                                                          realrobot=args.realrobot)
    
    # Load test trajectories
    if args.realrobot:
        seen_theta_test_trajs = np.load(args.test_trajs_set_path)
        test_trajs = seen_theta_test_trajs
    else:
        seen_theta_test_trajs_set_path = "../config/data_split_config/frankarobot_obj100_sg50_persg50/seen_theta_test_trajs_set_100_{}.npy".format(args.test_seed)
        seen_theta_test_trajs = np.load(seen_theta_test_trajs_set_path)
        test_trajs = seen_theta_test_trajs

    if args.preprocessed_traj_data is None:
        if args.realrobot:
            args.preprocessed_traj_data = os.path.join(params["irl"]["data_split_config_path"], f"preprocessed_traj_data_{args.seed}.pkl")
        else:
            args.preprocessed_traj_data = os.path.join(params["irl"]["data_split_config_path"], "test_seed_{}".format(args.test_seed), f"preprocessed_traj_data_{args.seed}.pkl")
    if not os.path.exists(os.path.dirname(args.preprocessed_traj_data)):
        os.makedirs(os.path.dirname(args.preprocessed_traj_data), exist_ok=True)

    print("Total trajectories:", len(all_trajs), "Train trajectories:", len(train_trajs), "Test trajectories:", len(test_trajs))
    traj_info = params["env"]["trajset_file"].split("/")[-1].split(".")[0]

    # Load previously saved demo indices for all thetas if the file exists, else use an empty dict.
    if os.path.exists(args.demo_indices_file):
        with open(args.demo_indices_file, "r") as f:
            saved_demo_indices = json.load(f)
            # cut demo_indices based on the number of demo_queries
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
    
    # Aggregate demos, features, and corresponding human thetas from multiple humans.
    with open(args.human_config, "r") as stream:
        humans_params_list = yaml.safe_load(stream)

    # train
    agg_demos = []
    agg_demo_features = []
    agg_demo_indices = []  # each demo's index in the original trajs
    agg_demo_states = []  # each demo's human state
    agg_demo_thetas = []  # each demo's human theta
    agg_train_trajs = []
    agg_train_features = [] 
    agg_train_states = []  # each training traj's human state
    agg_train_thetas = []  # each training traj's human theta

    # finetune
    agg_finetune_demos = []
    agg_finetune_demo_features = []
    agg_finetune_demo_states = []  # each finetune demo's human state
    agg_finetune_demo_thetas = []  # each finetune demo's human theta

    # Initialize lists to store win rates for unseen and seen human thetas.
    unseen_human_win_rates = []
    seen_theta_human_win_rates = []
    
    print("Make humans to get demo indices...")

    human_indices_split = json.load(open(human_indices_split_path, "r"))
    human_indices_split_train = human_indices_split["train_thetas"][:args.num_train_thetas]
    human_indices_split_test = human_indices_split["test_thetas"]
    human_indices_split_train_theta_keys = [get_theta_key(theta) for theta in human_indices_split_train]
    human_indices_split_test_theta_keys = [get_theta_key(theta) for theta in human_indices_split_test]
    counter = 0
    counter_train = 0
    counter_test = 0

    for human_idx, human_params in tqdm(enumerate(humans_params_list["humans"])):
        # Set environment type for the human
        human_params["type"] = params["env"]["type"]
        theta = human_params["preferencer"]["theta"]
        theta_key = get_theta_key(theta)

        if "featurized_trajs" not in preprocessed_traj_data:
            human = make_human(human_params, env, train_trajs)
            preprocessed_traj_data = {
                "featurized_trajs": human.featurized_trajs,
                "probs": human.probs,
            }
        else:
            human = make_human(human_params, env, train_trajs, **preprocessed_traj_data, scaling_coeffs=scaling_coeffs)

        # save
        if not os.path.exists(scaling_coeffs_path):
            joblib.dump(human.scaling_coeffs, scaling_coeffs_path)
            print(f"Saved scaling coefficients to {scaling_coeffs_path}")
        
        # Check if saved demo indices for this theta exist in the single JSON file.
        if theta_key in saved_demo_indices:
            demo_indices = saved_demo_indices[theta_key]
            if args.realrobot:
                # Map demo indices from all trajs to demo indices in train traj
                train_indices = train_test_split["train_indices"]
                demo_indices = [train_indices.index(idx) for idx in demo_indices if idx in train_indices]
            # Reconstruct demos using the saved indices (assuming demos are from train_trajs)
            demos = [train_trajs[i] for i in demo_indices]
            print(f"Loaded saved demo indices for theta {theta} with key {theta_key}")
        else:
            if not hasattr(human, 'preferencer'):
                human.set_trajset(train_trajs)
                human.set_preference(human_params["preferencer"])
            demos, demo_indices = human.generate_demos(args.demo_queries)
            saved_demo_indices[theta_key] = demo_indices
            print(f"Generated and saved demo indices for theta {theta} with key {theta_key}")

        if theta_key in human_indices_split_train_theta_keys:
            counter_train += 1
            demo_feats = human.featurized_trajs[demo_indices]
            demo_states = [human.calc_eef_object_states(traj, state_dim=args.state_dim) for traj in demos]

            train_feats = human.featurized_trajs
            train_states = [human.calc_eef_object_states(traj, state_dim=args.state_dim) for traj in train_trajs]
            human_theta = theta
            demo_thetas = [human_theta for _ in range(len(demos))]
            train_thetas = [human_theta for _ in range(len(train_trajs))]

            if 'explicitmask' in params["irl"]["type"]:
                if "llm_mask" in params["irl"]["type"]:
                    state_mask = theta_to_llm_state_mask(human_theta, state_dim=args.state_dim, llm_state_mask_path=args.llm_state_mask_path, language_ambiguity=args.language_ambiguity, llm_disambiguation=args.llm_disambiguation)
                else:
                    state_mask = theta_to_state_mask(human_theta, state_dim=args.state_dim)
                train_states = train_states * state_mask
                demo_states = demo_states * state_mask
            
            agg_demos.extend(demos)
            agg_demo_features.extend(demo_feats)
            agg_demo_states.extend(demo_states)
            agg_demo_thetas.extend(demo_thetas)
            agg_demo_indices.extend(demo_indices)
            agg_train_trajs.extend(train_trajs)
            agg_train_features.extend(train_feats)
            agg_train_states.extend(train_states)
            agg_train_thetas.extend(train_thetas)

            seen_theta_human_win_rates.append({
                "theta": human_theta,
                "demos": demos,
            })
        
        elif theta_key in human_indices_split_test_theta_keys:
            counter_test += 1
            human_theta = theta
            demo_feats = human.featurized_trajs[demo_indices]
            demo_states = [human.calc_eef_object_states(traj, state_dim=args.state_dim) for traj in demos]
            train_feats = human.featurized_trajs
            train_states = [human.calc_eef_object_states(traj, state_dim=args.state_dim) for traj in train_trajs]
            demo_thetas = [human_theta for _ in range(len(demos))]
            train_thetas = [human_theta for _ in range(len(train_trajs))]

            if 'explicitmask' in params["irl"]["type"]:
                if "llm_mask" in params["irl"]["type"]:
                    state_mask = theta_to_llm_state_mask(human_theta, state_dim=args.state_dim, llm_state_mask_path=args.llm_state_mask_path, language_ambiguity=args.language_ambiguity, llm_disambiguation=args.llm_disambiguation)
                else:
                    state_mask = theta_to_state_mask(human_theta, state_dim=args.state_dim)
                train_states = train_states * state_mask
                demo_states = demo_states * state_mask

            agg_finetune_demos.extend(demos)
            agg_finetune_demo_features.extend(demo_feats)
            agg_finetune_demo_states.extend(demo_states)
            agg_finetune_demo_thetas.extend(demo_thetas)
            
            unseen_human_win_rates.append({
                "theta": human_theta,
                "demos": demos,
            })
        else:
            counter += 1
            print(f"Human index {human_idx} not in train or test split. Skipping...")

        if "test_features" not in preprocessed_traj_data:
            if args.realrobot:
                test_features = [human.calc_features(traj) for traj in test_trajs]
                test_states = [human.calc_eef_object_states(traj, state_dim=args.state_dim) for traj in test_trajs]
            else:
                test_trajs = test_trajs.reshape(-1, *test_trajs.shape[2:])
                test_features = [human.calc_features(traj) for traj in test_trajs]
                test_states = [human.calc_eef_object_states(traj, state_dim=args.state_dim) for traj in test_trajs]
                test_trajs = test_trajs.reshape(-1, 2, *test_trajs.shape[1:])
            test_features = np.array(test_features)
            test_states = np.array(test_states)
            preprocessed_traj_data["test_features"] = test_features
            preprocessed_traj_data["test_states"] = test_states
        
    print(f"Total humans processed: {len(humans_params_list['humans'])}, Train humans: {counter_train}, Test humans: {counter_test}, Other humans: {counter}")
    
    # Save updated demo indices for all thetas back to the single JSON file.
    if not os.path.exists(args.demo_indices_file):
        with open(args.demo_indices_file, "w") as f:
            json.dump(saved_demo_indices, f, indent=4, default=jsonNpEncoder)
        print(f"Saved all demo indices to {args.demo_indices_file}")

    # Save preprocessed trajectory data to a file.
    if not os.path.exists(args.preprocessed_traj_data):
        with open(args.preprocessed_traj_data, "wb") as f:
            joblib.dump(preprocessed_traj_data, f)
        print(f"Saved preprocessed trajectory data to {args.preprocessed_traj_data}")
    
    train_features = np.array(agg_train_features)
    train_states = np.array(agg_train_states)
    
    # wandb
    dataset = params["env"]["trajset_file"].split("/")[-1].split(".")[0]
    state_dim = train_states.shape[-1]
    
    # Build wandb_run_name (used for model path even if wandb is disabled)
    if 'masked' in params['irl']['type']:
        wandb_run_name = f"{params['irl']['type']}_th{args.num_train_thetas}_d{args.demo_queries}_{dataset}_sdim{state_dim}_film0_it{params['irl']['num_iterations']}_lr{params['irl']['lr']}_hidden{'_'.join(map(str, params['irl']['hidden_sizes']))}_mweight{params['irl']['masked_loss_weight']}_mnoise{params['irl']['masked_loss_noise']}"
    else:
        wandb_run_name = f"{params['irl']['type']}_th{args.num_train_thetas}_d{args.demo_queries}_{dataset}_sdim{state_dim}_film0_it{params['irl']['num_iterations']}_lr{params['irl']['lr']}_hidden{'_'.join(map(str, params['irl']['hidden_sizes']))}"
    wandb_run_name = f"{wandb_run_name}_lr{params['irl']['lr']:.0e}_b{args.batch_size}_it{params['irl']['num_iterations']}"
    if 'llm' in params["irl"]["type"] and (args.llm_state_mask_path is not None):
        if "sdim" not in args.llm_state_mask_path.split("_")[-1]:
            wandb_run_name = wandb_run_name.replace("llm_mask", "llm_mask_" + args.llm_state_mask_path.split("_")[-1])
    if args.language_ambiguity is not None:
        wandb_run_name += "_lang_" + args.language_ambiguity
    else:
        wandb_run_name += "_lang_clear"
    if args.llm_disambiguation == "vlm":
        wandb_run_name += "_disambig_vlm"
    elif args.llm_disambiguation == "llm":
        wandb_run_name += "_disambig_llm"
    else:
        if 'llm' in params["irl"]["type"] and (args.llm_state_mask_path is not None):
            wandb_run_name += "_nodisambig"
    wandb_run_name += "_0914"
    if args.llm_state_mask_path is not None and "mixed" in args.llm_state_mask_path:
        wandb_run_name += "_mixed"
    
    if args.wandb:
        wandb.login(key=[line.strip() for line in open(os.path.expanduser("~/.wandb_api_key"))][0])
        if args.realrobot:
            wandb.init(project='Masked-IRL-realrobot', config=params, name=wandb_run_name, reinit=True)
        else:
            wandb.init(project='Masked-IRL-finetune-sdim19-0908-language-ambiguity-3train-3test', config=params, name=wandb_run_name, reinit=True)
        wandb.config.update({
            "seed": args.seed,
            "test_seed": args.test_seed,
            "demo_queries": args.demo_queries,
            "env_config": params,
            "args": vars(args),
            "state_dim": state_dim,
            "dataset": dataset,
            "llm_state_mask_path": args.llm_state_mask_path,
            "language_ambiguity": args.language_ambiguity,
            "llm_disambiguation": args.llm_disambiguation,
            "realrobot": args.realrobot,
        })
    
    hidden_sizes = params["irl"]["hidden_sizes"]
    hidden_sizes_str = "_".join(map(str, hidden_sizes))
    model_path = os.path.join("../data/models/irl", traj_info, wandb_run_name, "seed{}".format(args.seed))
    if args.debug:
        model_path = model_path + "_debug"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    noise_path = "../config/data_split_config/noise_200_21_9.npy"
    noise = np.load(noise_path)

    # Create a single IRL model with aggregated demos, features, and human thetas.
    irl = make_irl(params["irl"], None, agg_demos, agg_train_trajs,
                   train_states=train_states,
                   demo_states=np.array(agg_demo_states),
                   demo_thetas=np.array(agg_demo_thetas),
                   demo_indices=np.array(agg_demo_indices),
                   train_thetas=np.array(agg_train_thetas),
                   wandb=args.wandb,
                   human_win_rates=unseen_human_win_rates,
                   seen_theta_human_win_rates=seen_theta_human_win_rates,
                   test_trajs=test_trajs,
                   save_path=model_path,
                   test_features=preprocessed_traj_data["test_features"],
                   test_states=preprocessed_traj_data["test_states"],
                   language_ambiguity=args.language_ambiguity,
                   unseen_humans=unseen_human_win_rates,
                   finetune_demo_features = np.array(agg_finetune_demo_features),
                   finetune_demo_states = np.array(agg_finetune_demo_states),
                   finetune_demo_thetas = np.array(agg_finetune_demo_thetas),
                   llm_state_mask_path=args.llm_state_mask_path,
                   llm_disambiguation=args.llm_disambiguation,
                   )
    
    # Training phase
    print("Starting training phase...")
    irl.train(params["irl"]["num_iterations"], save_model=True)
    
    # Finetuning phase
    print("Starting finetuning phase...")
    finetune_iterations = 100 if args.realrobot else 1000
    finetune_losses = irl.finetune(finetune_iterations=finetune_iterations, start_iteration=params["irl"]["num_iterations"], save_model=True)
    
    # Final evaluation
    print("Performing final evaluation...")
    if args.wandb:
        final_results, final_avg_win_rate = irl.evaluate(irl.human_win_rates)
        final_seen_theta_results, final_seen_theta_avg_win_rate = irl.evaluate(irl.seen_theta_human_win_rates)
        
        wandb.log({
            "final_eval/unseen_theta_avg_win_rate": final_avg_win_rate,
            "final_eval/seen_theta_avg_win_rate": final_seen_theta_avg_win_rate,
        })
        
        print(f"Final evaluation - Unseen theta avg win rate: {final_avg_win_rate * 100:.2f}%")
        print(f"Final evaluation - Seen theta avg win rate: {final_seen_theta_avg_win_rate * 100:.2f}%")

    if args.wandb:
        wandb.finish()
        print("WANDB FINISHED")
