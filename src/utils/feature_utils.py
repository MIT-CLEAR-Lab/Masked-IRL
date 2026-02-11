import numpy as np
import json
import random

def get_theta_key(theta):
    """
    Generate an interpretable key for a given theta vector (assumed to be 5-dimensional with elements in {-1, 0, 1}).
    For example, a theta of [-1, 0, 1, 1, 0] will yield the key "-1_0_1_1_0".
    """
    return '_'.join(str(x) for x in theta)

def theta_to_language(human_thetas, language_ambiguity=None, llm_disambiguation=False, demo_idx=0, llm_state_mask_path=None):
    """
    Convert human thetas to language instructions.

    Args:
        human_thetas: List of human theta vectors.

    Returns:
        List of language instructions corresponding to each theta.
    """
    # cost = theta*feature, reward = -theta*feature
    # language instructions
    # table: feature low means table is close and high means table is far
    # laptop: feature low means laptop is far and high means laptop is close
    # proxemics: feature low means human face is far and high means human face is close
    # human: feature low means human is far and high means human is close
    # coffee: feature low means coffee is upright and high means coffee is upside down
    # order: table, human, laptop, proxemics, coffee
    
    if llm_disambiguation:
        if language_ambiguity == "omit_referent":
            ambiguous_instructions_template = [
                ["Stay away.", "Stay close."],
                ["Stay close.", "Stay away."],
                ["Stay close.", "Stay away."],
                ["Stay close.", "Stay away."],
                ["Keep it upside down.", "Keep it upright."]
            ]
            if llm_state_mask_path is None:
                if llm_disambiguation == "llm":
                    llm_state_mask_path = "../config/data_split_config/theta_to_language_and_mask_sdim19_llm.json"
                elif llm_disambiguation == "vlm":
                    llm_state_mask_path = "../config/data_split_config/theta_to_language_and_mask_sdim19.json"
                else:
                    raise ValueError("llm_disambiguation should be either 'llm' or 'vlm'")
            llm_state_mask = json.load(open(llm_state_mask_path, "r"))
            # lang_instructions = [llm_state_mask[get_theta_key(theta)]["omit_referent"]["disambiguated_instruction"][(demo_idx)] for theta in human_thetas]
            lang_instructions = []
            for theta in human_thetas:
                theta_key = get_theta_key(theta)
                if len(llm_state_mask[theta_key]["omit_referent"]["disambiguated_instruction"]) > 0:
                    lang_instructions.append(llm_state_mask[theta_key]["omit_referent"]["disambiguated_instruction"][(demo_idx)])
                else:
                    print("USED AMBIGUOUS INSTRUCTION FOR THETA KEY:", theta_key)
                    # just use ambiguous language if disambiguated instruction is not available
                    # lang_instructions.append(ambiguous_llm_state_mask[theta_key]["omit_referent"]["ambiguous_instruction"])
                    instruction = ""
                    for i, feature_instr in enumerate(ambiguous_instructions_template):
                        if abs(theta[i]) == 1: # later, we can make this to float
                            if theta[i] == -1:
                                instruction += feature_instr[0] + " "
                            else:
                                instruction += feature_instr[1] + " "
                    lang_instructions.append(instruction)
        elif language_ambiguity == "omit_expression":
            # llm_state_mask_path = "../config/data_split_config/theta_to_language_and_mask_sdim19.json"
            ambiguous_instructions_template = [
                ["table.", "table."],
                ["human.", "human."],
                ["laptop.", "laptop."],
                ["human face.", "human face."],
                ["coffee.", "coffee."]
            ]
            if llm_state_mask_path is None:
                if llm_disambiguation == "llm":
                    llm_state_mask_path = "../config/data_split_config/theta_to_language_and_mask_sdim19_llm.json"
                elif llm_disambiguation == "vlm":
                    llm_state_mask_path = "../config/data_split_config/theta_to_language_and_mask_sdim19.json"
                else:
                    raise ValueError("llm_disambiguation should be either 'llm' or 'vlm'")
            llm_state_mask = json.load(open(llm_state_mask_path, "r"))
            # lang_instructions = [llm_state_mask[get_theta_key(theta)]["omit_expression"]["disambiguated_instruction"][(demo_idx)] for theta in human_thetas]
            lang_instructions = []
            for theta in human_thetas:
                theta_key = get_theta_key(theta)
                if len(llm_state_mask[theta_key]["omit_expression"]["disambiguated_instruction"]) > 0:
                    lang_instructions.append(llm_state_mask[theta_key]["omit_expression"]["disambiguated_instruction"][(demo_idx)])
                else:
                    # just use ambiguous language if disambiguated instruction is not available
                    # lang_instructions.append(ambiguous_llm_state_mask[theta_key]["omit_expression"]["ambiguous_instruction"])
                    print("USED AMBIGUOUS INSTRUCTION FOR THETA KEY:", theta_key)
                    instruction = ""
                    for i, feature_instr in enumerate(ambiguous_instructions_template):
                        if abs(theta[i]) == 1: # later, we can make this to float
                            if theta[i] == -1:
                                instruction += feature_instr[0] + " "
                            else:
                                instruction += feature_instr[1] + " "
                    lang_instructions.append(instruction)
        else:
            raise ValueError("llm_disambiguation is True but language_ambiguity is None")
        return lang_instructions

    if language_ambiguity is None:
        instructions_template = [
            # order is feature low (theta -1), feature high (theta 1)
            # if theta is 0, then we don't care about that feature
            ["Stay away from the table surface.", "Stay close to the table surface."],
            ["Stay close to the human.", "Stay away from the human."],
            ["Stay close to the laptop.", "Stay away from the laptop."],
            ["Stay close to the human face.", "Stay away from the human face."],
            ["Keep the cup upside down.", "Keep the cup upright."]
        ]

    elif language_ambiguity == "omit_referent":
        instructions_template = [
            ["Stay away.", "Stay close."],
            ["Stay close.", "Stay away."],
            ["Stay close.", "Stay away."],
            ["Stay close.", "Stay away."],
            ["Keep it upside down.", "Keep it upright."]
        ]
    
    elif language_ambiguity == "omit_expression":
        instructions_template = [
            ["table.", "table."],
            ["human.", "human."],
            ["laptop.", "laptop."],
            ["human face.", "human face."],
            ["coffee.", "coffee."]
        ]
        
    elif language_ambiguity == "paraphrase":
        paraphrased_language_path = "../config/data_split_config/language_paraphrased_temp0.7_12345.json"
        paraphrased_language_dict = json.load(open(paraphrased_language_path, "r"))
        lang_instructions = [random.choice(paraphrased_language_dict[get_theta_key(theta)]) for theta in human_thetas]
        return lang_instructions
        
    else:
        raise ValueError


    # for each traj, make a language instruction based on human_theta
    lang_instructions = []
    for traj_idx in range(len(human_thetas)):
        instruction = ""
        for i, feature_instr in enumerate(instructions_template):
            if abs(human_thetas[traj_idx][i]) == 1: # later, we can make this to float
                if human_thetas[traj_idx][i] == -1:
                    instruction += feature_instr[0] + " "
                else:
                    instruction += feature_instr[1] + " "
        lang_instructions.append(instruction)
    
    return lang_instructions

def theta_to_state_mask(thetas, threshold=0.1, state_dim=9):
    """
    Convert one or more human theta vectors (each of length 5) into binary state masks for a 13-dimensional state.
    
    Mapping (example):
    - Dimensions 0-2 (EEF pos)
    - Dimensions 3 (EEF rot z)
    - Dimension 4 (table height): controlled by theta[0].
    - Dimensions 5-6 (human xy pos): controlled by theta[2].
    - Dimensions 7-8 (laptop pos): controlled by theta[1].
    
    Note: Theta indices for other features (e.g. coffee) are ignored in this mapping.
    
    Args:
        thetas: A list or numpy array of theta vectors (each of length 5) or a single theta vector.
        threshold: Threshold to decide binary inclusion.
        
    Returns:
        mask: If a single theta vector is provided, returns a numpy array of shape (13,).
              If multiple theta vectors are provided, returns a numpy array of shape (N, 13),
              where N is the number of theta vectors.
    """
    # Ensure thetas is a numpy array.
    thetas = np.array(thetas)
    # change to absolute values
    thetas = np.abs(thetas)
    # If a single theta vector is provided, reshape to (1, 5)
    if thetas.ndim == 1:
        thetas = thetas.reshape(1, -1)
    
    masks = []
    for theta in thetas:
        mask = np.zeros(state_dim)
        if state_dim == 9:
            # thetas: table, human, laptop, proxemics, coffee
            if theta[0] > threshold:
                # eef z and table height
                mask[2] = 1
                mask[4] = 1
            if theta[1] > threshold:
                # eef xy and human xy
                mask[0:2] = 1
                mask[5:7] = 1
            if theta[2] > threshold:
                # eef xy and laptop xy
                mask[0:2] = 1
                mask[7:9] = 1
            if theta[3] > threshold:
                # eef xy and human xy
                mask[0:2] = 1
                mask[5:7] = 1
            if theta[4] > threshold:
                # eef rot
                mask[3] = 1
        elif state_dim == 19:
            # thetas: table, human, laptop, proxemics, coffee
            if theta[0] > threshold:
                # eef z and table height
                mask[2] = 1
                mask[18] = 1
            if theta[1] > threshold:
                # eef xy and human xy
                mask[0:2] = 1
                mask[12:14] = 1
            if theta[2] > threshold:
                # eef xy and laptop xy
                mask[0:2] = 1
                mask[15:17] = 1
            if theta[3] > threshold:
                # eef xy and human xy
                mask[0:2] = 1
                mask[12:14] = 1
            if theta[4] > threshold:
                # eef rot
                mask[9] = 1 # zx
        elif state_dim == 11:
            # thetas: table, human, laptop, proxemics, coffee
            if theta[0] > threshold:
                # eef z and table height
                mask[2] = 1
                mask[10] = 1
            if theta[1] > threshold:
                # eef xy and human xy
                mask[0:2] = 1
                mask[4:6] = 1
            if theta[2] > threshold:
                # eef xy and laptop xy
                mask[0:2] = 1
                mask[7:9] = 1
            if theta[3] > threshold:
                # eef xy and human xy
                mask[0:2] = 1
                mask[4:6] = 1
            if theta[4] > threshold:
                # eef rot
                mask[3] = 1
        masks.append(mask)
    
    return np.array(masks)

def theta_to_llm_state_mask(thetas, language_ambiguity=None, demo_idx=0, state_dim=9, llm_state_mask_path=None, llm_disambiguation=False):
    # Ensure thetas is a numpy array.
    thetas = np.array(thetas)

    # If a single theta vector is provided, reshape to (1, 5)
    if thetas.ndim == 1:
        thetas = thetas.reshape(1, -1)

    if llm_disambiguation:
        ambiguous_llm_state_mask_path = "../config/data_split_config/theta_ambiguous_instr_to_pred_mask_sdim19.json"
        with open(ambiguous_llm_state_mask_path, "r") as f:
            ambiguous_llm_state_mask = json.load(f)
        if llm_state_mask_path is None:
            if llm_disambiguation == "llm":
                llm_state_mask_path = "../config/data_split_config/theta_to_language_and_mask_sdim19_llm.json"
            elif llm_disambiguation == "vlm":
                llm_state_mask_path = "../config/data_split_config/theta_to_language_and_mask_sdim19.json"
            else:
                raise ValueError("llm_disambiguation should be either 'llm' or 'vlm'")
        if language_ambiguity == "omit_referent":
            # use the disambiguated omit referent instruction to predict mask
            
            with open(llm_state_mask_path, "r") as f:
                llm_state_mask = json.load(f)
            masks = []
            # masks = [llm_state_mask[get_theta_key(theta)]["omit_referent"]["pred_state_masks"][(demo_idx)] for theta in thetas]
            for theta in thetas:
                theta_key = get_theta_key(theta)
                if len(llm_state_mask[theta_key]["omit_referent"]["pred_state_masks"]) > 0:
                    masks.append(llm_state_mask[theta_key]["omit_referent"]["pred_state_masks"][(demo_idx)])
                else:
                    # just use ambiguous language mask if disambiguated mask is not available
                    masks.append(ambiguous_llm_state_mask[theta_key]["omit_referent"]["pred_state_mask"])
                    
        elif language_ambiguity == "omit_expression":
            # use the disambiguated omit expression instruction to predict mask
            llm_state_mask_path = "../config/data_split_config/theta_to_language_and_mask_sdim19.json"
            with open(llm_state_mask_path, "r") as f:
                llm_state_mask = json.load(f)
            # masks = [llm_state_mask[get_theta_key(theta)]["omit_expression"]["pred_state_masks"][(demo_idx)] for theta in thetas]
            masks = []
            for theta in thetas:
                theta_key = get_theta_key(theta)
                if len(llm_state_mask[theta_key]["omit_expression"]["pred_state_masks"]) > 0:
                    masks.append(llm_state_mask[theta_key]["omit_expression"]["pred_state_masks"][(demo_idx)])
                else:
                    # just use ambiguous language mask if disambiguated mask is not available
                    masks.append(ambiguous_llm_state_mask[theta_key]["omit_expression"]["pred_state_mask"])
        else:
            raise ValueError("llm_disambiguation is True but language_ambiguity is not recognized")
        # raise NotImplementedError("llm_disambiguation is not implemented for theta_to_llm_state_mask")
    else:
        if language_ambiguity is None:
            # clear instruction
            if llm_state_mask_path is None:
                llm_state_mask_path = "../config/data_split_config/theta_to_pred_mask_sdim{}.json".format(state_dim)
            with open(llm_state_mask_path, "r") as f:
                llm_state_mask = json.load(f)
            # return the mask for thetas
            masks = [llm_state_mask[get_theta_key(theta)] for theta in thetas]
        elif language_ambiguity in ["omit_referent", "omit_expression"]:
            # use the ambiguous instruction to predict mask
            llm_state_mask_path = "../config/data_split_config/theta_ambiguous_instr_to_pred_mask_sdim19.json"
            with open(llm_state_mask_path, "r") as f:
                llm_state_mask = json.load(f)
            masks = [llm_state_mask[get_theta_key(theta)][language_ambiguity]["pred_state_mask"] for theta in thetas]

    return np.array(masks)

def theta_to_reward_density(thetas):
    # dense: valid feature 4, 5
    # medium: valid feature 3
    # sparse: valid feature 1, 2
    # count the number of nonzero values in each theta and return the list of densities
    # use matrix operations for efficiency
    densities = np.count_nonzero(thetas, axis=1)
    # convert to list of strings - sparse, medium, dense
    # also use map function for brevity
    density_labels = list(map(lambda s: "sparse" if s <= 2 else "medium" if s == 3 else "dense", densities))
    return density_labels