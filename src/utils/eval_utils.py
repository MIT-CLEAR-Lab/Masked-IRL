import cProfile
import io
import os
import pstats

import numpy as np
from scipy import special
from scipy.spatial import distance
import random

from src.utils.feature_utils import theta_to_reward_density

def calc_metrics(pref, trajs, metric_vals):
    return {"Test Featurizer Accuracy": pref.featurizer.last_test_accuracy,
            "Test Preference Accuracy": pref.last_test_accuracy}


def profile(fnc):
    """A decorator that uses cProfile to profile a function (from https://osf.io/upav8/)"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())

        filename = os.path.expanduser(os.path.join("~", fnc.__name__ + ".pstat"))
        print(filename)
        pr.dump_stats(filename)

        return retval

    return inner

def calculate_win_rate(test_trajs, ground_truth_reward, learned_reward, test_features=None, test_states=None):
    """
    Calculate the win rate by comparing preference labels from ground truth and learned rewards.
    """
    win_count = 0
    num_samples = 100
    traj_pairs = [(test_trajs[i], test_trajs[j], test_features[i], test_features[j], test_states[i], test_states[j])
                  for i in range(len(test_trajs)) for j in range(i+1, len(test_trajs))]
    traj_pairs = random.sample(traj_pairs, num_samples)
    total_count = len(traj_pairs)
    # for traj1, traj2 in traj_pairs:
    for traj1, traj2, feat1, feat2, state1, state2 in traj_pairs:
        gt_reward_1 = ground_truth_reward(feat1)
        gt_reward_2 = ground_truth_reward(feat2)
        gt_preference = gt_reward_1 > gt_reward_2

        lr_reward_1 = learned_reward(traj1, state1)
        lr_reward_2 = learned_reward(traj2, state2)
        lr_preference = lr_reward_1 > lr_reward_2

        if gt_preference == lr_preference:
            win_count += 1
    return win_count / total_count

def count_num_valid_features(theta):
    # Count the number of non-zero features in the theta vector. valid features are those that are not zero. consider both positive and negative values.
    num_valid_features = np.sum(np.abs(theta) > 0)
    return num_valid_features

def count_avg_win_rate_per_num_valid_features(human_win_rates, results):
    result_dict = {}
    result_dict_per_reward_density = {}
    for i, info in enumerate(human_win_rates):
        num_valid_features = count_num_valid_features(info['theta'])
        reward_density = theta_to_reward_density([info['theta']])[0]
        if num_valid_features not in result_dict:
            result_dict[num_valid_features] = []
        if reward_density not in result_dict_per_reward_density:
            result_dict_per_reward_density[reward_density] = []
        result_dict[num_valid_features].append(results[i])
        result_dict_per_reward_density[reward_density].append(results[i])
    for num_valid_features in result_dict:
        result_dict[num_valid_features] = np.mean(result_dict[num_valid_features])
    for reward_density in result_dict_per_reward_density:
        result_dict_per_reward_density[reward_density] = np.mean(result_dict_per_reward_density[reward_density])
    return result_dict, result_dict_per_reward_density