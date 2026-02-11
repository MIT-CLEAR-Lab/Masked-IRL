import os
import numpy as np
import torch
import tqdm
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

from src.models.mlp import MLP

class MEIRL:
    def __init__(self, params, featurizer, demos, all_trajs, train_features=None, demo_features=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.featurizer = featurizer
        if self.featurizer is not None:
            if self.featurizer.feat_scale_construct is not None:
                self.featurizer.feat_scale_construct(all_trajs) # creates feature scaling
        self.orig_demos = np.array(demos)
        # self.demos = self.featurizer.featurize(demos)
        self.demos = demo_features
        self.orig_trajs = np.array(all_trajs)
        # self.all_trajs = self.featurizer.featurize(all_trajs) # minyoung - fix
        # self.featurized_all_trajs = self.featurizer.featurize(all_trajs)
        self.train_features = train_features
        self.all_trajs = train_features

        # send all data to device
        self.demos = torch.as_tensor(self.demos).to(self.device)
        self.all_trajs = torch.as_tensor(self.all_trajs).to(self.device)

        input_dim = self.demos.shape[1]
        self.lr = params["lr"]
        
        if params["linear"]:
            self.cost_nn = MLP(input_dim, 1, [], output_activation=None).to(self.device)
        else:
            self.cost_nn = MLP(input_dim, 1, params["hidden_sizes"], output_activation='softplus').to(self.device)
        self.optimizer = optim.Adam(self.cost_nn.parameters(), lr=params["lr"])

    def print_weights(self):
        for name, param in self.cost_nn.named_parameters():
            if param.requires_grad:
                print (name, param.data)

    def calc_cost(self, trajs, trajs_features=None):
        # breakpoint()
        if not torch.is_tensor(trajs):
            trajs = torch.as_tensor(trajs).to(self.device)
        if trajs_features is not None:
            if not torch.is_tensor(trajs_features):
                trajs_features = torch.as_tensor(trajs_features).to(self.device)
            cost = self.cost_nn(trajs_features)
        else:
            cost = self.cost_nn(self.featurizer.featurize(trajs))
        return cost.squeeze(1).detach().cpu().numpy()

    def train(self, iterations, save_dir=None, save_path=None, save_loss=False):
        losses = []
        p = tqdm.tqdm(range(iterations))
        for epoch in p:
            self.optimizer.zero_grad()

            # samples trajectory according to current cost distribution (importance sampling as in GCL)
            rand_indices = torch.multinomial(F.softmax(-self.cost_nn(self.all_trajs).squeeze(1)), len(self.demos), replacement=True)
            samples = self.all_trajs[rand_indices]
            # importance weights
            probs_demos = F.softmax(-self.cost_nn(self.demos).squeeze(1).detach())
            probs_samples = F.softmax(-self.cost_nn(samples).squeeze(1).detach())

            cost_demos = self.cost_nn(self.demos)
            cost_samples = self.cost_nn(samples)

            loss = torch.mean(cost_demos) + torch.log(torch.mean(torch.exp(-cost_samples)/probs_samples) + torch.mean(torch.exp(-cost_demos)/probs_demos))
            loss.backward()
            self.optimizer.step()

            print('Epoch %d loss: %.8f' %(epoch + 1, loss.item()))
            losses.append(loss.item())

        if save_dir is not None or save_path is not None:
            if save_path is None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(self.cost_nn, os.path.join(save_dir, 'cost_network_it{}_lr{}.pt'.format(iterations, self.lr)))
            else:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(self.cost_nn, save_path)
            if save_loss:
                plt.plot(losses)
                plt.legend(['Total loss'])
                plt.savefig(os.path.join(save_dir, 'losses_it{}_lr{}.png'.format(iterations, self.lr)))
                plt.close()
        return losses