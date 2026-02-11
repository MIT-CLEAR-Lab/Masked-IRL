import os
import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import argparse
import yaml

from src.utils.parser import *
from src.utils.bullet_utils import *
from src.utils.feature_utils import theta_to_language

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and replay trajectory subsets with feature-based filtering.")
    parser.add_argument("--traj-path", default="../data/traj_sets/frankarobot_obj20_sg10_persg5.npy")
    parser.add_argument("--save-path", default="../data/images/trajset.png")
    parser.add_argument("--env-config", default="../config/reward_learning/obj20_sg10_persg5/maskedrl.yaml")
    parser.add_argument("--feature", default="table", help="Feature name to rank trajectories by.")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--show-top", action="store_true", default=True)
    parser.add_argument("--show-worst", action="store_true", default=False)
    parser.add_argument("--replay-forever", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--capture-image", action="store_true", help="If set, capture a rendered image to save_path.")
    parser.add_argument("--show-ee-coords", dest="show_ee_coords", default=False, action="store_true")
    parser.add_argument("--debug", action="store_true", help="Drop into ipdb at end.")
    parser.add_argument("--show-demos", action="store_true", default=False)
    args = parser.parse_args()

    if args.show_worst:
        args.show_top = False

    traj_path = args.traj_path
    print("Trajectory set loaded from", traj_path)
    traj_set = np.load(traj_path, allow_pickle=True)
    print("Trajectory set shape", traj_set.shape)

    env_config = args.env_config
    with open(env_config, "r") as stream:
        params = yaml.safe_load(stream)

    set_seed(args.seed)
    env = make_env(params["env"])

    all_trajs, all_trajs, test_trajs = load_split_data(
        traj_path,
        params["env"]["per_SG"],
        params["env"]["train_test_split"]
    )
    print(len(all_trajs), len(test_trajs), len(all_trajs))

    humans_params = [
        {
            "features": ["table", "human", "laptop", "proxemics", "coffee"],
            "feature_scaling": "normalize",
            "preferencer": {
                "theta": [1,0,0,0,0],
                "beta": 20.0,
                "f_method": "boltzmann",
                "s_method": "luce"
            }
        }
    ]

    # - theta*features is reward. theta*features is cost.
    human_params = humans_params[-1]
    human_params["type"] = params["env"]["type"]
    print(human_params)
    human = make_human(human_params, env, all_trajs)

    all_features = np.array([human.calc_features(traj) for traj in all_trajs])
    print(all_features.shape)

    trajs = all_trajs
    objectID = env.objectID

    save_path = args.save_path
    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cam = p.getDebugVisualizerCamera()

    valid_feature = args.feature
    feature_idx = human.features.index(valid_feature)
    features = all_features[:, feature_idx]

    top_n = args.n # n trajectories most preferred when theta is 1
    worst_n = args.n # n trajectories most preferred when theta is -1

    # lowest feature value means most preferred when theta is 1
    top_n_idx = np.argsort(features)[:top_n] if top_n > 0 else []
    top_n_trajs = trajs[top_n_idx] if top_n > 0 else []
    top_theta = np.zeros(len(human.features))
    top_theta[feature_idx] = 1.0
    top_language_instruction = theta_to_language([top_theta])[0]
    print("Feature:", valid_feature)
    print("Top theta:", top_theta)
    print("Top n demo indices:", top_n_idx)
    print("Top language instruction:", top_language_instruction)
    print("="*20)

    # highest feature value means most preferred when theta is -1
    worst_n_idx = np.argsort(features)[-worst_n:] if worst_n > 0 else []
    worst_n_trajs = trajs[worst_n_idx] if worst_n > 0 else []
    worst_theta = -top_theta
    worst_language_instruction = theta_to_language([worst_theta])[0]
    print("Worst theta:", worst_theta)
    print("Worst n demo indices:", worst_n_idx)
    print("Worst language instruction:", worst_language_instruction)

    if args.show_ee_coords or "coffee" in valid_feature:
        show_ee_coords = True
    else:
        show_ee_coords = False

    replay_forever = args.replay_forever
    if args.show_top and len(top_n_trajs):
        if not args.show_demos:
            print("Showing top {} trajectories...".format(len(top_n_trajs)))
            env.replay(
                top_n_trajs,
                colors=[[0, 0, 1] for _ in range(len(top_n_trajs))],
                forever=replay_forever,
                show_ee_coords=show_ee_coords
            )
        else:
            human_params["preferencer"]["theta"] = top_theta.tolist()
            human = make_human(human_params, env, all_trajs)
            demos, demo_indices = human.generate_demos(10)
            print("Top Sampled Demo indices from Boltzmann Human:", demo_indices)
            env.replay(
                np.concatenate([top_n_trajs, demos], axis=0),
                colors=[[0, 0, 1] for _ in range(len(top_n_trajs))] + [[1, 1, 1] for _ in range(len(demos))],
                forever=replay_forever,
                show_ee_coords=show_ee_coords
            )
        
    if args.show_worst and len(worst_n_trajs):
        if not args.show_demos:
            print("Showing worst {} trajectories...".format(len(worst_n_trajs)))
            env.replay(
                worst_n_trajs,
                colors=[[1, 0, 0] for _ in range(len(worst_n_trajs))],
                forever=replay_forever,
                show_ee_coords=show_ee_coords
            )
        else:
            human_params["preferencer"]["theta"] = worst_theta.tolist()
            human = make_human(human_params, env, all_trajs)
            demos, demo_indices = human.generate_demos(10)
            print("Worst Sampled Demo indices from Boltzmann Human:", demo_indices)
            env.replay(
                np.concatenate([worst_n_trajs, demos], axis=0),
                colors=[[1, 0, 0] for _ in range(len(worst_n_trajs))] + [[1, 1, 1] for _ in range(len(demos))],
                forever=replay_forever,
                show_ee_coords=show_ee_coords
            )

    if args.capture_image and save_path:
        view_matrix = p.computeViewMatrix(cameraEyePosition=[2, -2, 3],
                                          cameraTargetPosition=[0, 0, 1],
                                          cameraUpVector=[0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=10)
        image = p.getCameraImage(640, 640, view_matrix, projection_matrix,
                                 shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        plt.imsave(save_path, image[2])
        print(f"Saved image to {save_path}")

    if args.debug:
        import ipdb; ipdb.set_trace()

    p.removeAllUserParameters()
    p.disconnect()