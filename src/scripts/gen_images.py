import os
import sys
sys.path.append("../")
sys.path.append("../../")
import yaml
from src.utils.parser import *
from src.utils.feature_utils import theta_to_language, theta_to_state_mask

config = "../config/reward_learning/obj20_sg10_persg5/maskedrl.yaml"
with open(config, "r") as stream:
    params = yaml.safe_load(stream)

params["env"]["real"]=True
env = make_env(params["env"])

traj_path = "../data/traj_sets/frankarobot_obj20_sg10_persg5.npy"
env.gen_traj_imgs(traj_path, camera_angles=3, save_loc='../data/traj_viz/')  