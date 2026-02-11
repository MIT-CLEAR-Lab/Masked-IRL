#!/usr/bin/env python3
"""
Generate trajectories for multiple object locations.

This script generates trajectories for different object location configurations,
combining them into a single dataset. For each location, it randomly samples
human_center, laptop_center, and table_center positions, then generates
trajectories using the standard trajectory generation script.
"""

import argparse
import os
import sys
import numpy as np
import random
import yaml
import subprocess

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.parser import set_seed


def generate_trajs_multiple_locations(
    config_path,
    num_obj_locations,
    nsamples,
    seed=0,
    save_dir="./data/traj_sets",
    tag=None,
    combine=True
):
    """
    Generate trajectories for multiple object locations.
    
    Args:
        config_path: Path to base config YAML file
        num_obj_locations: Number of different object location configurations
        nsamples: Number of start-goal pairs per location
        seed: Random seed
        save_dir: Directory to save trajectories
        tag: Optional tag for the generated trajectory set
        combine: If True, combine all trajectories into a single array; 
                 if False, save as dictionary with location info
    """
    # Load base config
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    
    # Set random seed
    set_seed(seed)
    random.seed(seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "tmp"), exist_ok=True)
    
    # Create temporary config directory
    tmp_config_dir = os.path.join(os.path.dirname(config_path), "tmp")
    os.makedirs(tmp_config_dir, exist_ok=True)
    
    # Determine final save path
    env_type = params["env"]["type"]
    per_SG = params["env"]["per_SG"]
    
    if tag is None:
        tag = f"obj{num_obj_locations}_sg{nsamples}_persg{per_SG}"
    
    if combine:
        final_save_path = os.path.join(save_dir, f"{env_type}_{tag}.npy")
    else:
        final_save_path = os.path.join(save_dir, f"{env_type}_{tag}_dict.npy")
    
    trajs_dict = {}
    
    print(f"Generating trajectories for {num_obj_locations} object locations...")
    print(f"Each location will have {nsamples} start-goal pairs with {per_SG} trajectories per pair")
    
    for i in range(num_obj_locations):
        print(f"\n--- Location {i+1}/{num_obj_locations} ---")
        
        # Randomly sample object centers
        human_center = [
            random.uniform(-1.2, -0.2),
            random.uniform(-1.0, -0.6),
            0.9
        ]
        table_center = [-0.65, 0.0, 0.0]
        laptop_center = [
            random.uniform(-1.0, -0.2),
            random.uniform(-0.3, 0.3),
            0.635 + table_center[2]
        ]
        
        object_centers = {
            'HUMAN_CENTER': human_center,
            'LAPTOP_CENTER': laptop_center,
            'TABLE_CENTER': table_center
        }
        
        print(f"  human_center: [{human_center[0]:.2f}, {human_center[1]:.2f}, {human_center[2]:.2f}]")
        print(f"  laptop_center: [{laptop_center[0]:.2f}, {laptop_center[1]:.2f}, {laptop_center[2]:.2f}]")
        print(f"  table_center: [{table_center[0]:.2f}, {table_center[1]:.2f}, {table_center[2]:.2f}]")
        
        # Update config with new object centers
        params["env"]["object_centers"] = object_centers
        
        # Save temporary config
        tmp_config_path = os.path.join(tmp_config_dir, f"tmp_config_obj{i}.yaml")
        with open(tmp_config_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        
        # Generate trajectories using the standard script
        obj_tag = f"obj{i}"
        tmp_save_path = os.path.join(save_dir, "tmp", f"{env_type}_{obj_tag}_sg{nsamples}_persg{per_SG}.npy")
        
        # Call generate_traj_sets.py
        script_path = os.path.join(os.path.dirname(__file__), "generate_traj_sets.py")
        # Get project root (assuming script is in src/scripts/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        cmd = [
            sys.executable,
            script_path,
            "--config", tmp_config_path,
            "--samples", str(nsamples),
            "--tag", obj_tag,
            "--save_dir", os.path.join(save_dir, "tmp")
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_root, env={**os.environ, "PYTHONPATH": project_root})
        
        if result.returncode != 0:
            print(f"  ERROR: Failed to generate trajectories for location {i}")
            continue
        
        # Load generated trajectories
        if not os.path.exists(tmp_save_path):
            # Try alternative path (script might have saved with different naming)
            alt_path = os.path.join(save_dir, "tmp", f"{env_type}_{obj_tag}_sg{nsamples}_persg{per_SG}.npy")
            if os.path.exists(alt_path):
                tmp_save_path = alt_path
            else:
                print(f"  ERROR: Generated trajectory file not found at {tmp_save_path}")
                continue
        
        trajs = np.load(tmp_save_path, allow_pickle=True)
        print(f"  Loaded {len(trajs)} trajectories")
        
        trajs_dict[i] = {
            "object_centers": object_centers,
            "trajs": trajs,
        }
        
        # Save intermediate results
        if not combine:
            np.save(final_save_path, trajs_dict)
            print(f"  Saved intermediate results to {final_save_path}")
    
    # Combine all trajectories if requested
    if combine:
        print(f"\n--- Combining trajectories ---")
        all_trajs = []
        for i in range(num_obj_locations):
            if i in trajs_dict:
                trajs = trajs_dict[i]["trajs"]
                all_trajs.extend(trajs)
        
        all_trajs = np.array(all_trajs)
        print(f"Total trajectories: {len(all_trajs)}")
        print(f"Trajectory shape: {all_trajs[0].shape if len(all_trajs) > 0 else 'N/A'}")
        
        np.save(final_save_path, all_trajs)
        print(f"Saved combined trajectories to {final_save_path}")
    else:
        print(f"\n--- Saving dictionary format ---")
        np.save(final_save_path, trajs_dict)
        print(f"Saved trajectory dictionary to {final_save_path}")
        print(f"Dictionary contains {len(trajs_dict)} locations")
    
    # Clean up temporary config files
    if os.path.exists(tmp_config_dir):
        for f in os.listdir(tmp_config_dir):
            if f.startswith("tmp_config_obj"):
                os.remove(os.path.join(tmp_config_dir, f))
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate trajectories for multiple object locations'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to base config YAML file'
    )
    parser.add_argument(
        '--num_obj_locations',
        type=int,
        required=True,
        help='Number of different object location configurations'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=50,
        help='Number of start-goal pairs per location (default: 50)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed (default: 0)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default="./data/traj_sets",
        help='Directory to save trajectories (default: ./data/traj_sets)'
    )
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help='Optional tag for the generated trajectory set'
    )
    parser.add_argument(
        '--no-combine',
        action='store_true',
        help='Save as dictionary with location info instead of combining into single array'
    )
    
    args = parser.parse_args()
    
    generate_trajs_multiple_locations(
        config_path=args.config,
        num_obj_locations=args.num_obj_locations,
        nsamples=args.samples,
        seed=args.seed,
        save_dir=args.save_dir,
        tag=args.tag,
        combine=not args.no_combine
    )
