import argparse
import os
import sys
import yaml

sys.path.insert(1, '../')

from src.utils.parser import *

if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('-c', '--config', type=str, help='config file', required=True)
    parser.add_argument('--save_dir', type=str, default="./data/traj_sets", help='save folder')
    parser.add_argument('--samples', type=int, default=10000, help='samples')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--tag', type=str, default=None, help='optional description for the generated trajectory set')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        params = yaml.safe_load(stream)

    # Set random seed.
    set_seed(args.seed)

    # Create environment and generate trajs.
    env = make_env(params["env"])
    trajs = env.generate_trajs(args.samples, per_SG=params["env"]["per_SG"])
    print(len(trajs))

    # Save trajectories to file.
    here = os.path.abspath(os.getcwd())
    if not os.path.exists(here + "/{}".format(args.save_dir)):
        os.makedirs(here + "/{}".format(args.save_dir))
    if args.tag is not None:
        np.save(here + "/{}/{}_{}_sg{}_persg{}.npy".format(args.save_dir, params["env"]["type"], args.tag, len(trajs)//params["env"]["per_SG"], params["env"]["per_SG"]), trajs)
    else:
        np.save(here + "/{}/{}_sg{}_persg{}.npy".format(args.save_dir, params["env"]["type"], len(trajs)//params["env"]["per_SG"], params["env"]["per_SG"]), trajs)
