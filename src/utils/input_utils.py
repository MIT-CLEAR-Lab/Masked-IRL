import torch
import numpy as np


def transform_input(x, trans_dict):
    """
    transforms x, TxNxD raw space input tensor to the desired input space tensor as specified in trans_dict.
    Input
        - TxNxD torch Tensor from environment
        - trans_dict with all settings
    Output:
        - Tx... torch Tensor according to settings
    """
    # Transform a torch input x according to noangles, noxyz, norot, EErot, 6D_laptop, 6D_human, 9D_coffee, lowdim parameters.

    if len(x.shape) == 2:
        x = torch.unsqueeze(x, axis=0)

    num_joints = 7
    if trans_dict['noxyz']:
        x = x[:, :, :70]

    # if lowdim, need to remove the joints 1 through 6.
    if trans_dict['lowdim']:
        if not trans_dict['noxyz']:
            x = torch.cat((x[:, :, :70], x[:, :, 70+(num_joints-1)*3:]), dim=2)
        x = torch.cat((x[:, :, :7], x[:, :, 7+(num_joints-1)*9:]), dim=2)
        num_joints = 1
    elif trans_dict['EErot']:
        x = torch.cat((x[:, :, :7], x[:, :, 7+(num_joints-1)*9:]), dim=2)
        num_joints = 1

    if trans_dict['norot']:
        x = torch.cat((x[:, :, :7], x[:, :, 7+num_joints*9:]), dim=2)

    if trans_dict['noangles']:
        x = x[:, :, 7:]

    return x


def create_sim_validation(all_trajs, human, device='cpu', samples=4000):
    # Create validation set
    anchor_test_states, positive_test_states, negative_test_states = [], [], []
    skippeda_test_states, skipped1_test_states, skipped2_test_states = [], [], []

    for i in range(samples):
        anchor_index, sample1_index, sample2_index = np.random.randint(0, len(all_trajs)), np.random.randint(0, len(all_trajs)), np.random.randint(0, len(all_trajs))
        while anchor_index == sample1_index or anchor_index == sample2_index or sample1_index == sample2_index:
            anchor_index, sample1_index, sample2_index = np.random.randint(0, len(all_trajs)), np.random.randint(0, len(all_trajs)), np.random.randint(0, len(all_trajs))
        anchor = all_trajs[anchor_index]
        sample1 = all_trajs[sample1_index]
        sample2 = all_trajs[sample2_index]
        response = human.query_triplet(anchor, sample1, sample2, validate=True)
        if response == "SKIPPED":
            skippeda_test_states.append(anchor)
            skipped1_test_states.append(sample1)
            skipped2_test_states.append(sample2)
        else:
            anchor, positive, negative = response
            anchor_test_states.append(anchor)
            positive_test_states.append(positive)
            negative_test_states.append(negative)
    anchor_test_states = torch.as_tensor(np.stack(anchor_test_states, axis=0)).to(device)
    positive_test_states = torch.as_tensor(np.stack(positive_test_states, axis=0)).to(device)
    negative_test_states = torch.as_tensor(np.stack(negative_test_states, axis=0)).to(device)
    if len(skippeda_test_states) != 0:
        skippeda_test_states = torch.as_tensor(np.stack(skippeda_test_states, axis=0)).to(device)
        skipped1_test_states = torch.as_tensor(np.stack(skipped1_test_states, axis=0)).to(device)
        skipped2_test_states = torch.as_tensor(np.stack(skipped2_test_states, axis=0)).to(device)

    return anchor_test_states, positive_test_states, negative_test_states, skippeda_test_states, skipped1_test_states, skipped2_test_states


def create_pref_validation(test_trajs, human, device='cpu', samples=4000):
    # Create validation set
    pref1_states, pref2_states, pref_labels = [], [], []

    for i in range(samples):
        pref1_index, pref2_index = np.random.randint(0, len(test_trajs)), np.random.randint(0, len(test_trajs))
        while pref1_index == pref2_index:
            pref1_index, pref2_index = np.random.randint(0, len(test_trajs)), np.random.randint(0, len(test_trajs))
        pref1_traj = test_trajs[pref1_index]
        pref2_traj = test_trajs[pref2_index]
        label = human.query_pref(pref1_traj, pref2_traj, validate=True)
        pref1_states.append(pref1_traj)
        pref2_states.append(pref2_traj)
        pref_labels.append(label)

    pref1_test_states = torch.as_tensor(np.stack(pref1_states, axis=0)).to(device)
    pref2_test_states = torch.as_tensor(np.stack(pref2_states, axis=0)).to(device)
    pref_labels = torch.as_tensor(np.stack(pref_labels, axis=0)).to(device)

    return pref1_test_states, pref2_test_states, pref_labels
