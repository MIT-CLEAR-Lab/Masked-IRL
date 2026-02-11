import random

import numpy as np
import torch
import time
import json
import os

from src.envs.gridrobot import Gridrobot
from src.envs.gridworld import Gridworld
from src.envs.jacorobot import Jacorobot
from src.envs.frankarobot import Frankarobot
from src.models.featurizers.identity import Identity
from src.models.featurizers.meta_preference import MetaPreference
from src.models.featurizers.random import Random
from src.models.featurizers.similarity import Similarity
from src.models.featurizers.vae import VAE
from src.models.humans.gridrobot_human import GridrobotHuman
from src.models.humans.gridworld_human import GridworldHuman
from src.models.humans.jacorobot_human import JacorobotHuman
from src.models.humans.frankarobot_human import FrankarobotHuman
from src.models.humans.frankarobot_human_real import FrankarobotHumanReal
from src.models.reward_learning.meirl_lang_states_input import MEIRL_Lang
from src.models.reward_learning.masked_rl_lang_states_input import MaskedRL
from src.models.reward_learning.masked_rl_lang_states_input_llm_mask import MaskedRL_LLM_Mask
from src.models.reward_learning.meirl_lang_states_input_gcl_full import MEIRL_Lang_GCL_FULL
from src.models.reward_learning.masked_rl_lang_states_input_gcl_full import MaskedRL_GCL_FULL
from src.models.reward_learning.masked_rl_lang_states_input_llm_mask_gcl_full import MaskedRL_LLM_Mask_GCL_FULL
from src.models.reward_learning.meirl_lang_states_input_no_lang import MEIRL_Lang_No_Lang
from src.models.reward_learning.masked_rl_lang_states_input_no_lang import MaskedRL_No_Lang
from src.models.reward_learning.preferences import Preferences
from src.models.evaluation.feature_learner import FeatureLearner


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# def load_split_data(trajset_file, per_SG, train_test_split=0.8):
#     all_trajs = np.load(trajset_file)
#     orig_shape = all_trajs[0].shape

#     all_trajs = all_trajs.reshape((-1, int(per_SG), *orig_shape))
#     np.random.shuffle(all_trajs)
#     all_trajs = all_trajs.reshape((-1, *orig_shape))

#     assert int(train_test_split * len(all_trajs)) % per_SG == 0
#     assert len(all_trajs) % per_SG == 0
#     train_trajs, test_trajs = all_trajs[:int(train_test_split * len(all_trajs))], all_trajs[int(train_test_split * len(all_trajs)):]
#     return all_trajs, train_trajs, test_trajs


def load_split_data(trajset_file, per_SG, train_test_split=0.8, indices_file=None):
    """
    Load trajectories and split into training and test sets using consistent indices.
    If the indices file exists, load the saved train and test indices.
    Otherwise, randomly shuffle and compute the indices, and save them (if indices_file is provided).
    """
    all_trajs = np.load(trajset_file)
    orig_shape = all_trajs[0].shape

    # Reshape to account for per_SG grouping
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

def make_env(params):
    if params["type"] == "gridworld":
        return Gridworld(params["X"], params["Y"], params["obstacles"], params["starts"], params["goals"])
    elif params["type"] == "gridrobot":
        return Gridrobot(params["X"], params["Y"], params["obstacles"], params["starts"], params["goals"], params["feat_obsts"])
    elif params["type"] == "jacorobot":
        return Jacorobot(params["object_centers"], params["resources_dir"], params["horizon"], params["timestep"], params["real"])
    elif params["type"] == "frankarobot":
        return Frankarobot(params["object_centers"], params["resources_dir"], params["horizon"], params["timestep"], params["real"])
    elif params["type"] == "frankarobot":
        return Frankarobot(params["object_centers"], params["resources_dir"], params["horizon"], params["timestep"], params["real"])
    elif params["type"] == "frankarobot_real":
        return Frankarobot(params["object_centers"], params["resources_dir"], params["horizon"], params["timestep"], params["real"])
    else:
        raise NotImplementedError


def make_human(params, env, trajs, **kwargs):
    start = time.time()
    if params["type"] == "gridworld":
        human = GridworldHuman(params, env)
    elif params["type"] == "gridrobot":
        human = GridrobotHuman(params, env)
    elif params["type"] == "jacorobot":
        human = JacorobotHuman(params, env)
    elif params["type"] == "frankarobot":
        human = FrankarobotHuman(params, env, **kwargs)
    elif params["type"] == "frankarobot_real":
        human = FrankarobotHumanReal(params, env, **kwargs)
    else:
        raise NotImplementedError
    print("human created in", time.time() - start)
    start = time.time()
    if not hasattr(human, "featurized_trajs"):
        human.set_trajset(trajs)
    elif not hasattr(human, "scaling_coeffs"):
        human.scaling_coeffs = []
        human.all_trajs = trajs
        human.feat_scale_construct(trajs)
    print("trajset set in", time.time() - start)
    start = time.time()
    if "preferencer" in params:
        if not hasattr(human, "probs"):
            human.set_preference(params["preferencer"])
    print("preferencer set in", time.time() - start)
    # print("human created in", time.time() - start)
    return human


def make_featurizer(params, env_params, test_humans):
    if "input_dict" in env_params:
        params["input_dict"] = env_params["input_dict"]
    params["trajectory_shape"] = np.array(test_humans[0].all_trajs).shape[1:]
    if params["type"] == "Identity":
        return Identity(params)
    elif params["type"] == "Random":
        return Random(params)
    elif params["type"] == "MetaPreference":
        return MetaPreference(params, test_humans)
    elif params["type"] == "VAE":
        return VAE(params)
    elif params["type"] == "Similarity":
        return Similarity(params, test_humans[0])
    # elif params["type"] == "Human":
    #     return 
    else:
        raise NotImplementedError


# def make_irl(params, featurizer, demos, all_trajs, all_demos=None, train_features=None, demo_features=None, **args):
def make_irl(params, featurizer, demos, all_trajs, all_demos=None, train_states=None, demo_states=None, **args):
# def make_irl(params, featurizer, demos, all_trajs, all_demos=None, train_features=None, demo_features=None, demo_masks=None, train_masks=None):
    if params["type"] == "meirl":
        return MEIRL(params, featurizer, demos, all_trajs, train_states, demo_states)
    elif params["type"] == "meirl_lang":
        return MEIRL_Lang(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "maskedrl":
        return MaskedRL(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "maskedrl_llm_mask":
        return MaskedRL_LLM_Mask(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "explicitmask_lang":
        return MEIRL_Lang(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "explicitmask_lang_llm_mask":
        return MEIRL_Lang(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    # GCL full versions
    elif params["type"] == "meirl_lang_gcl_full":
        return MEIRL_Lang_GCL_FULL(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "maskedrl_gcl_full":
        return MaskedRL_GCL_FULL(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "maskedrl_llm_mask_gcl_full":
        return MaskedRL_LLM_Mask_GCL_FULL(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "explicitmask_lang_gcl_full":
        return MEIRL_Lang_GCL_FULL(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "explicitmask_lang_llm_mask_gcl_full":
        return MEIRL_Lang_GCL_FULL(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    # single model, no language conditioned baselines
    elif params["type"] == "meirl_lang_no_lang":
        return MEIRL_Lang_No_Lang(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "maskedrl_no_lang":
        return MaskedRL_No_Lang(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    elif params["type"] == "explicitmask_lang_no_lang":
        return MEIRL_Lang_No_Lang(params, featurizer, demos, all_trajs, train_states, demo_states, **args)
    else:
        raise NotImplementedError


def make_preferences(params, featurizer, human):
    if params["type"] == "preferences":
        return Preferences(params, featurizer, human)
    else:
        raise NotImplementedError


def make_evaluator(params, featurizer, human):
    if params["type"] == "feature_learner":
        return FeatureLearner(params, featurizer, human)
    else:
        raise NotImplementedError
