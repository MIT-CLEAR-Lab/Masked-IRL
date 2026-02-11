import sys
import numpy as np
import random
import torch
import os
from torch import nn
from torch import optim
from scipy import special
from src.utils import input_utils

import wandb

sys.path.insert(1, '../')

from src.models.mlp import MLP


class PrefFrameNet(nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.net = MLP(2, 1, hidden_layers, dropout=0.1)

    def forward(self, x):
        # B x L -> B x L/2 x 2 -> (B * L/2) x 2 -> (B * L/2) x 1 -> B x L/2 -> B x 1
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 2)
        rew = self.net.forward(x)
        return torch.sum(rew.reshape(batch_size, -1), 1).unsqueeze(1)


class PrefTrajNet(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        self.net = MLP(input_dim, 1, hidden_layers)

    def forward(self, x):
        return self.net.forward(x)


class Preferences:
    def __init__(self, params, featurizer, human):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.params = params

        self.featurizer = featurizer
        self.human = human
        traj_dim = self.featurizer.traj_dim
        input_dim = self.featurizer.featurize(np.empty(traj_dim, dtype=np.float32)).shape[1]
        if params["use_traj"]:
            self.rew_nns = [PrefTrajNet(input_dim, params["hidden_layers"]).to(self.device) for _ in range(params["num_ensemble"])]
        else:
            self.rew_nns = [PrefFrameNet(params["hidden_layers"]).to(self.device) for _ in range(params["num_ensemble"])]

        self.pref_trajs1 = torch.empty(traj_dim).to(self.device)
        self.pref_trajs2 = torch.empty(traj_dim).to(self.device)
        self.pref_labels = torch.empty((0, )).to(self.device)
        self.num_finetunes = 0

        self.last_test_accuracy = None

    def calc_rewards(self, trajs, index=None, train=False):
        if not torch.is_tensor(trajs):
            trajs = torch.as_tensor(trajs).to(self.device)
        if index is not None:
            reward = self.rew_nns[index](self.featurizer.featurize(trajs, index=index, train=self.params["train_featurizer"]))
        else:
            reward = torch.mean(torch.stack([rew_nn(self.featurizer.featurize(trajs, index=i, train=self.params["train_featurizer"]))
                                             for i, rew_nn in enumerate(self.rew_nns)], dim=0), dim=0)
        reward = torch.clamp(reward.squeeze(1), -50, 50)
        if train:
            return reward
        return reward.detach().cpu().numpy()

    def calc_distribution(self, states):
        with torch.no_grad():
            if self.params["per_SG"] == 1:
                pair_indices = [random.sample(range(len(states)), k=2) for _ in range(self.params["num_generated_pairs"])]
            else:
                pair_indices = np.array([np.random.choice(self.params["per_SG"], 2, replace=False) for _ in range(self.params["num_generated_pairs"])]) + \
                                  self.params["per_SG"] * np.random.choice(len(states) // self.params["per_SG"], size=(self.params["num_generated_pairs"],))[:, None]

            if not torch.is_tensor(states):
                states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
            # reward_trajs = [self.calc_rewards(states, index=i, train=False) for i in range(len(self.rew_nns))]

            all_distributions = []
            for triplet_index in pair_indices:
                distributions = []
                for i in range(len(self.rew_nns)):
                    sampled_rew1 = self.calc_rewards(states[triplet_index[0]][None, ...])[0]
                    sampled_rew2 = self.calc_rewards(states[triplet_index[1]][None, ...])[0]
                    probs = special.softmax([sampled_rew1, sampled_rew2])
                    distributions.append(probs)
                all_distributions.append(distributions)
            all_distributions = np.array(all_distributions)
            return all_distributions, np.array(pair_indices)

    def gen_query(self, all_trajs):
        all_distributions, pair_indices = self.calc_distribution(all_trajs)
        mean = np.mean(all_distributions, axis=1)

        if self.params["query_technique"] == "random":
            score = np.random.random((len(all_distributions),))
        elif self.params["query_technique"] == "infogain":
            mean_entropy = -(mean[:, 0] * np.log2(mean[:, 0]) + mean[:, 1] * np.log2(mean[:, 1]))

            ind_entropy = np.zeros_like(mean_entropy)
            for i in range(all_distributions.shape[1]):
                ind_entropy += -(all_distributions[:, i, 0] * np.log2(all_distributions[:, i, 0] + 1e-5) +
                                 all_distributions[:, i, 1] * np.log2(all_distributions[:, i, 1] + 1e-5))
            score = mean_entropy - ind_entropy / all_distributions.shape[1]
            wandb.log({"Misc/Pref/Query Score": np.max(score), "Misc/Pref/Query Mean Entropy Score": mean_entropy[np.argmax(score)],
                       "Misc/Pref/Query Ind Entropy Score": (ind_entropy / all_distributions.shape[1])[np.argmax(score)]})
        else:
            raise NotImplementedError

        num_queries = min(self.params["query_batch_size"], self.params["num_pref_queries"] - self.human.num_pref_queries)
        indices = np.argpartition(score, -num_queries)[-num_queries:]
        return zip(pair_indices[indices], score[indices])

    def loss(self, all_trajs):
        if self.human.num_pref_queries < self.params["num_pref_queries"] and self.num_finetunes % self.params["steps_between_queries"] == 0:
            for (traj1_index, traj2_index), kl_max in self.gen_query(all_trajs):
                traj1 = all_trajs[traj1_index]
                traj2 = all_trajs[traj2_index]
                label = self.human.query_pref(traj1, traj2)
                self.pref_trajs1 = torch.cat([self.pref_trajs1, torch.as_tensor(traj1).to(self.device).unsqueeze(0)], dim=0)
                self.pref_trajs2 = torch.cat([self.pref_trajs2, torch.as_tensor(traj2).to(self.device).unsqueeze(0)], dim=0)
                self.pref_labels = torch.cat([self.pref_labels, torch.as_tensor([label]).to(self.device)], dim=0)
            assert self.human.num_pref_queries <= self.params["num_pref_queries"]

        losses = []
        if len(self.pref_trajs1) > 0:
            self.num_finetunes += 1
            indices = np.random.choice(range(len(self.pref_trajs1)), size=min(self.params["batch_size"], len(self.pref_trajs1)), replace=False)
            indices = torch.as_tensor(indices).to(self.device)
            pref_trajs1_batch = torch.index_select(self.pref_trajs1, 0, indices)
            pref_trajs2_batch = torch.index_select(self.pref_trajs2, 0, indices)
            pref_labels_batch = torch.index_select(self.pref_labels, 0, indices)

            for i, rew_nn in enumerate(self.rew_nns):
                rew1, rew2 = self.calc_rewards(pref_trajs1_batch, index=i, train=True), self.calc_rewards(pref_trajs2_batch, index=i, train=True)
                rew_probs = torch.softmax(torch.stack([rew1, rew2], dim=1), dim=1)
                loss = (1 - pref_labels_batch) * rew_probs[:, 0] + pref_labels_batch * rew_probs[:, 1]
                losses.append(-torch.mean(loss) + self.params["regularization"] * torch.mean(torch.square(rew1) + torch.square(rew2)))
        return losses

    def train(self, iterations, trajs, test_trajs=None, start_log=True, pref_log_counter=0):
        if start_log:
            env_name = type(self.human).__name__.replace("Human", "")
            wandb.init(project="SIRL", entity="yiliu77", group="Preferences", dir="../data/", config=self.params, name=env_name + "_Pref_" + self.featurizer.name)
        opts = [optim.Adam(list(self.rew_nns[i].parameters()) + list(self.featurizer.parameters()) if self.params["train_featurizer"]
                           else self.rew_nns[i].parameters(), lr=self.params["lr"]) for i in range(len(self.rew_nns))]
        print(self.params["train_featurizer"])

        if test_trajs is not None and not hasattr(self, 'test_states'):
            # Create validation set
            self.human.reset()
            self.test_states = input_utils.create_pref_validation(test_trajs, self.human, device=self.device)

        for i in range(iterations):
            losses = self.loss(trajs)

            for j in range(len(opts)):
                if len(losses) > 0:
                    opts[j].zero_grad()
                    losses[j].backward()
                    opts[j].step()

        with torch.no_grad():
            for i in range(20):
                losses = self.loss(trajs)
                log_results = {}
                avg_results = {"Avg Pref Train Accuracy ({})".format(str(self.human)): 0, "Avg Pref Test Accuracy ({})".format(str(self.human)): 0}
                for j in range(len(opts)):
                    log_results["Pref{}/Pref Loss".format(j)] = losses[j].item()

                    if len(self.pref_trajs1) > 0:
                        pref1_rew = self.calc_rewards(self.pref_trajs1, index=j, train=False)
                        pref2_rew = self.calc_rewards(self.pref_trajs2, index=j, train=False)
                        pref_labels_np = self.pref_labels.detach().cpu().numpy()
                        indices = np.argwhere(pref_labels_np != 0.5)
                        train_accuracy = np.mean(np.equal(np.argmax(np.stack([pref1_rew, pref2_rew], axis=1), axis=1)[indices], pref_labels_np[indices]).astype(np.float32))
                        train_accuracy = train_accuracy if len(indices) > 0 else 0.5
                        log_results["Pref{}/Pref Train Accuracy ({})".format(j, str(self.human))] = train_accuracy
                        avg_results["Avg Pref Train Accuracy ({})".format(str(self.human))] += train_accuracy

                    # Validation accuracy of triplets
                    if test_trajs is not None:
                        pref1_test_states, pref2_test_states, pref_test_labels = self.test_states
                        pref1_rew = self.calc_rewards(pref1_test_states, index=j, train=False)
                        pref2_rew = self.calc_rewards(pref2_test_states, index=j, train=False)
                        pref_test_labels_np = pref_test_labels.cpu().numpy()
                        indices = np.argwhere(pref_test_labels_np != 0.5)
                        test_accuracy = np.mean(np.equal(np.argmax(np.stack([pref1_rew, pref2_rew], axis=1), axis=1)[indices], pref_test_labels_np[indices]).astype(np.float32))
                        test_accuracy = test_accuracy if len(indices) > 0 else 0.5
                        log_results["Pref{}/Pref Test Accuracy ({})".format(j, str(self.human))] = test_accuracy
                        avg_results["Avg Pref Test Accuracy ({})".format(str(self.human))] += test_accuracy

                        if self.last_test_accuracy is None:
                            self.last_test_accuracy = test_accuracy.item()
                        self.last_test_accuracy = self.last_test_accuracy * 0.9 + test_accuracy * 0.1

                for k, v in avg_results.items():
                    log_results[k] = v / len(losses)
                log_results["Pref Log Counter"] = pref_log_counter
                wandb.log(log_results)
                pref_log_counter += 1

        return pref_log_counter

    def save_model(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for i in range(len(self.rew_nns)):
            torch.save(self.rew_nns[i].state_dict(), folder_name + '/reward_nn{}'.format(i))
        np.save(folder_name + '/test_accuracy', np.array(self.last_test_accuracy))

    def load_model(self, folder_name):
        for i in range(len(self.rew_nns)):
            self.rew_nns[i].load_state_dict(torch.load(folder_name + '/reward_nn{}'.format(i), map_location=torch.device(self.device)))
        self.last_test_accuracy = np.load(folder_name + '/test_accuracy.npy')
