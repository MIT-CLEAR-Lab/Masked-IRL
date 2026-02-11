import os
import numpy as np
import torch
import tqdm
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import wandb
import time

from src.models.mlp import MLP
# Import the function that converts human theta to language instructions.
from src.utils.feature_utils import theta_to_language, theta_to_state_mask, theta_to_reward_density
from src.utils.eval_utils import calculate_win_rate, count_num_valid_features, count_avg_win_rate_per_num_valid_features

#####################
# Language Encoder  #
#####################
class SimpleLanguageEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, output_dim):
        """
        A simple language encoder using an embedding layer and a linear layer.
        This serves as an option for the language encoder.
        """
        super(SimpleLanguageEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim, output_dim)
    
    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        emb = self.embedding(token_ids)  # (batch_size, seq_len, emb_dim)
        # Mean-pool over sequence length.
        emb = emb.mean(dim=1)
        out = self.linear(emb)  # (batch_size, output_dim)
        return out

# Optionally, one can create wrappers for BERT or T5 encoders using HuggingFace transformers.
# For instance:
class BertLanguageEncoder(nn.Module):
    def __init__(self, output_dim):
        super(BertLanguageEncoder, self).__init__()
        from transformers import BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Add a linear layer to map BERT's hidden size to output_dim.
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)
    
    def forward(self, instructions):
        # instructions: a list of strings.
        encoded = self.tokenizer(instructions, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(next(self.bert.parameters()).device)
        attention_mask = encoded["attention_mask"].to(next(self.bert.parameters()).device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        out = self.linear(cls_output)
        return out

# Similarly, one could define a T5 encoder. Here we show only the simple and bert options.
class T5LanguageEncoder(nn.Module):
    def __init__(self, output_dim):
        super(T5LanguageEncoder, self).__init__()
        from transformers import T5EncoderModel, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.encoder = T5EncoderModel.from_pretrained("t5-base")
        # Freeze encoder if desired, e.g.:
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.linear = nn.Linear(self.encoder.config.d_model, output_dim)
    
    def forward(self, instructions):
        """
        instructions: list of strings.
        """
        encoded = self.tokenizer(instructions, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(next(self.encoder.parameters()).device)
        attention_mask = encoded["attention_mask"].to(next(self.encoder.parameters()).device)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over token embeddings.
        pooled = outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
        out = self.linear(pooled)  # (batch_size, output_dim)
        return out

#####################
# FiLM Reward Model #
#####################
class FiLMBlock(nn.Module):
    def __init__(self, state_dim, cond_dim):
        """
        A FiLM block that modulates states based on a conditioning vector.
        """
        super(FiLMBlock, self).__init__()
        self.scale = nn.Linear(cond_dim, state_dim)
        self.shift = nn.Linear(cond_dim, state_dim)
    
    def forward(self, states, cond):
        gamma = self.scale(cond)
        beta = self.shift(cond)
        return gamma * states + beta

class FiLMRewardModel(nn.Module):
    def __init__(self, state_dim, cond_dim, hidden_sizes):
        """
        A reward model that conditions on state states and language embedding (condition)
        using FiLM layers.
        """
        super(FiLMRewardModel, self).__init__()
        self.film0 = FiLMBlock(state_dim, cond_dim)
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        # self.film1 = FiLMBlock(hidden_sizes[0], cond_dim)
        layers = []
        in_dim = hidden_sizes[0]
        for h in hidden_sizes[1:]:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.fc_layers = nn.Sequential(*layers)
        # self.film_final = FiLMBlock(in_dim, cond_dim)
        self.out = nn.Linear(in_dim, 1)
    
    def forward(self, state_states, cond):
        x = self.film0(state_states, cond)
        x = F.relu(self.fc1(x))
        # x = self.film1(x, cond)
        x = self.fc_layers(x)
        # x = self.film_final(x, cond)
        reward = self.out(x)
        return reward

##############################
# MaskedRL: Main Class       #
##############################
class MEIRL_Lang:
    def __init__(self, params, featurizer, demos, all_trajs, 
                 train_states=None, demo_states=None, 
                 demo_thetas=None, train_thetas=None, wandb=False, human_win_rates=None, seen_theta_human_win_rates=None, 
                 test_trajs=None, save_path=None,
                 test_features=None, test_states=None, 
                 use_state_encoder=False, 
                 unseen_humans=None, finetune_demo_features=None, finetune_demo_states=None, finetune_demo_thetas=None,
                 language_ambiguity=None, llm_disambiguation=False, llm_state_mask_path=None,
                 **kwargs):
        """
        Reward model conditioned on language embeddings derived from human thetas.
        
        Args:
            params: configuration dictionary.
            featurizer: feature extractor for states.
            demos: raw demonstration trajectories.
            all_trajs: all trajectories for training IRL.
            train_states: states for training trajectories.
            demo_states: states for demonstration trajectories.
            demo_thetas: list/array of human theta vectors (one per demonstration).
            train_thetas: list/array of human theta vectors for training trajectories.
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.featurizer = featurizer
        if self.featurizer is not None and hasattr(self.featurizer, 'feat_scale_construct'):
            self.featurizer.feat_scale_construct(all_trajs)
        self.orig_demos = np.array(demos)
        self.demos = demo_states  # demonstration states
        self.orig_trajs = np.array(all_trajs)
        self.train_states = train_states
        self.all_trajs = train_states
        self.finetune_demo_states = torch.as_tensor(finetune_demo_states).to(self.device)

        self.demos = torch.as_tensor(self.demos).to(self.device)
        self.all_trajs = torch.as_tensor(self.all_trajs).to(self.device)

        if demo_thetas is not None:
            self.demo_thetas = torch.as_tensor(demo_thetas).float().to(self.device)
        else:
            raise ValueError("demo_thetas must be provided.")
        if train_thetas is not None:
            self.train_thetas = torch.as_tensor(train_thetas).float().to(self.device)
        else:
            raise ValueError("train_thetas must be provided.")

        
        # Convert human thetas to language instructions.
        # theta_to_language returns a list of instructions.
        self.language_ambiguity=language_ambiguity
        self.llm_disambiguation = llm_disambiguation
        self.llm_state_mask_path = llm_state_mask_path
        self.demo_language_instructions = theta_to_language(demo_thetas, self.language_ambiguity, llm_disambiguation=self.llm_disambiguation, llm_state_mask_path=self.llm_state_mask_path)
        self.train_language_instructions = theta_to_language(train_thetas, self.language_ambiguity, llm_disambiguation=self.llm_disambiguation, llm_state_mask_path=self.llm_state_mask_path)
        self.finetune_demo_language_instructions = theta_to_language(finetune_demo_thetas, self.language_ambiguity, llm_disambiguation=self.llm_disambiguation, llm_state_mask_path=self.llm_state_mask_path)
                # Choose a language encoder option.
        # Options: "simple", "bert". Default to "simple".
        encoder_choice = params["language_encoder"] if "language_encoder" in params else "simple"
        vocab_size = params["language"].get("vocab_size", 10000)
        emb_dim = params["language"].get("emb_dim", 128)
        # We want the final language embedding to have dimension equal to theta_dim.
        # theta_dim = self.demo_thetas.shape[1]
        if encoder_choice == "simple":
            self.lang_encoder = SimpleLanguageEncoder(vocab_size, emb_dim, theta_dim).to(self.device)
        elif encoder_choice == "bert":
            self.lang_encoder = BertLanguageEncoder(emb_dim).to(self.device)
        elif encoder_choice == "t5":
            self.lang_encoder = T5LanguageEncoder(emb_dim).to(self.device)
        else:
            raise ValueError(f"Unsupported language encoder: {encoder_choice}")

        # For FiLM conditioning, we now use a FiLM reward network.
        self.use_state_encoder = use_state_encoder
        # state_dim = self.demos.shape[1]
        
        state_dim = self.demos.shape[-1]
        self.state_dim = state_dim
        hidden_sizes = params.get("hidden_sizes", [128, 128, 128])
        self.lr = params["lr"]
        self.batch_size = params.get("batch_size", 64)
        film_cond_dim = emb_dim
        self.cost_nn = FiLMRewardModel(state_dim=state_dim, cond_dim=film_cond_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.cost_nn = self.cost_nn.to(torch.float64)
        # self.optimizer = optim.Adam(self.cost_nn.parameters(), lr=params["lr"])
        # optimize the optimizer for the language encoder as well
        self.optimizer = optim.Adam(list(self.cost_nn.parameters()) + list(self.lang_encoder.parameters()), lr=self.lr)
        self.wandb = wandb
        self.human_win_rates = human_win_rates
        self.seen_theta_human_win_rates = seen_theta_human_win_rates
        self.test_trajs = test_trajs
        self.test_features = test_features
        self.test_states = test_states
        self.finetune_demo_features = finetune_demo_features
        self.finetune_demo_thetas = finetune_demo_thetas
        self.unseen_humans = unseen_humans
        self.save_path = save_path
        if self.wandb:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            self.last_ckpt = os.path.join(self.save_path, 'irl_last.pt')
            self.best_ckpt = os.path.join(self.save_path, 'irl_best.pt')
            self.finetune_last_ckpt = os.path.join(self.save_path, 'irl_finetune_last.pt')
            self.finetune_best_ckpt = os.path.join(self.save_path, 'irl_finetune_best.pt')

    def tokenize_instructions(self, instructions):
        """
        A simple tokenizer that splits by whitespace and converts words to indices.
        For production, you may replace this with a proper tokenizer.
        """
        tokenized = []
        for instr in instructions:
            tokens = [hash(word) % 10000 for word in instr.split()]  # simple hash-based
            tokenized.append(tokens)
        max_len = max(len(t) for t in tokenized)
        token_tensor = []
        for t in tokenized:
            padded = t + [0]*(max_len - len(t))
            token_tensor.append(padded)
        token_tensor = torch.tensor(token_tensor).to(self.device)
        return token_tensor

    def print_weights(self):
        for name, param in self.cost_nn.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def calc_traj_cost(self, traj, lang_emb):
        """
        Compute the aggregated cost for a trajectory.
        
        Args:
            traj: A 2D numpy array or tensor of shape (T, state_dim), where T is the number of states in the trajectory.
            lang_emb: A language embedding vector of shape (cond_dim,) corresponding to this trajectory.
        
        Returns:
            Aggregated cost (e.g., sum of per-state costs).
        """
        # Convert traj to tensor if necessary
        if not torch.is_tensor(traj):
            traj = torch.as_tensor(traj).to(self.device)
        # Expand the language embedding to match the number of states
        lang_emb = lang_emb.unsqueeze(0).repeat(traj.shape[0], 1)
        # Compute cost for each state
        cost_per_state = self.cost_nn(traj, lang_emb)
        # Aggregate per-state cost (e.g., sum)
        return torch.sum(cost_per_state).detach().cpu().numpy()

    def calc_cost(self, trajs, trajs_states=None, lang_embs=None):
        costs = []
        # If precomputed states are not provided, compute them from the featurizer.
        if trajs_states is None:
            trajs_states = [self.featurizer.featurize(traj) for traj in trajs]
        for i, state_seq in enumerate(trajs_states):
            # Ensure lang_embs[i] is a tensor of the correct shape.
            cost = self.calc_traj_cost(state_seq, lang_embs[i])
            costs.append(cost)
        return np.array(costs)

    def calc_traj_cost_batch(self, traj_batch, lang_emb_batch):
        """
        Compute the aggregated cost for each trajectory in a mini-batch in a fully vectorized manner.
        
        Args:
            traj_batch: Tensor of shape (B, T, state_dim) representing a batch of trajectories.
            lang_emb_batch: Tensor of shape (B, cond_dim) representing the language embedding for each trajectory.
            
        Returns:
            traj_cost: Tensor of shape (B,) containing the aggregated cost (sum over T) for each trajectory.
        """
        B, T, state_dim = traj_batch.shape
        # Expand language embedding for each state.
        lang_emb_expanded = lang_emb_batch.unsqueeze(1).expand(-1, T, -1)  # (B, T, cond_dim)
        # Flatten trajectories and corresponding language embeddings.
        traj_flat = traj_batch.view(B * T, state_dim)
        lang_flat = lang_emb_expanded.reshape(B * T, -1)
        cost_flat = self.cost_nn(traj_flat, lang_flat)  # (B*T, 1)
        cost_flat = cost_flat.view(B, T)
        # Aggregate per-trajectory cost (e.g., sum over T)
        traj_cost = torch.sum(cost_flat, dim=1)  # (B,)
        return traj_cost

    def evaluate(self, human_win_rates, num_samples=100, return_rewards=False):
        """
        Evaluate the model by computing avg win rate across all humans using vectorized operations.
        Args:
            test_features: numpy array of shape (N, feature_dim)
            test_states: list or array of shape (N, T, state_dim)
            num_samples: number of random trajectory pairs to sample per human
        Returns:
            Average win rate across humans
        """
        # Precompute tensor of test states
        states_tensor = torch.as_tensor(self.test_states, device=self.device)
        N = self.test_features.shape[0]
        # Vectorized GT rewards: for each human
        win_rates = []
        gt_rewards_list = []
        lr_rewards_list = []
        for info in human_win_rates:
            theta = info['theta']
            # GT rewards vectorized
            gt_rewards = np.dot(self.test_features, theta)
            # reshape into -1, 2
            gt_rewards = gt_rewards.reshape(-1, 2)
            # Learned rewards vectorized
            inst = theta_to_language([theta], self.language_ambiguity, llm_disambiguation=self.llm_disambiguation, llm_state_mask_path=self.llm_state_mask_path)[0]
            with torch.no_grad():
                emb = self.lang_encoder(inst).to(torch.float64)
                emb_batch = emb.expand(N, -1)
                lr_rewards = -self.calc_traj_cost_batch(states_tensor, emb_batch).detach().cpu().numpy()
            # reshape into -1, 2
            lr_rewards = lr_rewards.reshape(-1, 2)
            # Sample random pairs
            # idx = np.random.choice(N, size=(num_samples, 2), replace=False)
            # prefs_gt = gt_rewards[idx[:,0]] > gt_rewards[idx[:,1]]
            # prefs_lr = lr_rewards[idx[:,0]] < lr_rewards[idx[:,1]] # reward is negative cost
            # gt_rewards and lr_rewards are already paired, so we can directly compare them.
            prefs_gt = gt_rewards[:, 0] > gt_rewards[:, 1]
            prefs_lr = lr_rewards[:, 0] < lr_rewards[:, 1]  # reward is negative cost
            win_rates.append((np.sum(prefs_gt == prefs_lr) / (N//2)))
            gt_rewards_list.append(gt_rewards)
            lr_rewards_list.append(lr_rewards)
        if not return_rewards:
            return win_rates, float(np.mean(win_rates))
        return win_rates, float(np.mean(win_rates)), gt_rewards_list, lr_rewards_list

    def evaluate_regret(self, human_win_rates):
        """
        Evaluate the regret of the learned reward function.
        
        Args:
            human_win_rates: List of dicts with 'theta' keys for each human.
        
        Returns:
            Average regret across humans.
        """
        states_tensor = torch.as_tensor(self.test_states, device=self.device)
        N = self.test_features.shape[0]
        regrets = []
        for info in human_win_rates:
            theta = info['theta']
            # GT rewards vectorized
            gt_rewards = np.dot(self.test_features, theta)
            # Learned rewards vectorized
            inst = theta_to_language([theta], self.language_ambiguity, llm_disambiguation=self.llm_disambiguation, llm_state_mask_path=self.llm_state_mask_path)[0]
            with torch.no_grad():
                emb = self.lang_encoder(inst).to(torch.float64)
                emb_batch = emb.expand(N, -1)
                lr_rewards = -self.calc_traj_cost_batch(states_tensor, emb_batch).detach().cpu().numpy()
            # Compute regret as the difference between the max GT reward and the GT reward of the trajectory with max learned reward.
            best_gt_reward = np.max(gt_rewards)
            best_lr_idx = np.argmax(lr_rewards)
            regret = best_gt_reward - gt_rewards[best_lr_idx]
            regrets.append(regret)
            
        return regrets, float(np.mean(regrets))

    def evaluate_reward_variance(self, human_win_rates, noise_level=1.0, perturb_valid=False):
        """
        Evaluate the variance in the learned reward function by adding noise to the test trajectories.

        Args:
            test_trajs: List of trajectories for evaluation.
            ground_truth_reward: Function to compute ground truth rewards.
            learned_reward: Function to compute learned rewards.
            noise_level: Standard deviation of Gaussian noise to add to trajectories.

        Prints:
            The variance in the learned reward outputs.
        """
        states_tensor = torch.as_tensor(self.test_states, device=self.device)
        N = self.test_features.shape[0]

        # calculate learned rewards for states_tensor and get the min and max value so that we can normalize the test reward values
        total_learned_rewards = []
        for info in human_win_rates:
            theta = info['theta']
            inst = theta_to_language([theta], self.language_ambiguity, llm_disambiguation=self.llm_disambiguation, llm_state_mask_path=self.llm_state_mask_path)[0]
            emb = self.lang_encoder(inst).to(torch.float64)
            emb_batch = emb.expand(N, -1)
            learned_rewards = -self.calc_traj_cost_batch(states_tensor, emb_batch).detach().cpu().numpy()
            total_learned_rewards.append(learned_rewards)
        learned_rewards_min = np.min(total_learned_rewards)
        learned_rewards_max = np.max(total_learned_rewards)
        reward_variance_list = []
        reward_variance_per_reward_densities = {"sparse": [], "medium": [], "dense": []}

        for info in human_win_rates:
            theta = info['theta']
            reward_density = theta_to_reward_density([theta])[0]

            state_mask = theta_to_state_mask(theta, state_dim=self.state_dim)
            if perturb_valid:
                noise_mask = state_mask
            else:
                noise_mask = 1 - state_mask

            learned_reward_variance_list = []
            
            n_repeat = 5
            inst = theta_to_language([theta], self.language_ambiguity, llm_disambiguation=self.llm_disambiguation, llm_state_mask_path=self.llm_state_mask_path)[0]
            with torch.no_grad():
                emb = self.lang_encoder(inst).to(torch.float64)
                emb_batch = emb.expand(N, -1)
                # lr_rewards = -self.calc_traj_cost_batch(states_tensor, emb_batch).detach().cpu().numpy()
                noisy_rewards = []
                for _ in range(n_repeat):
                    traj_noise = np.random.normal(0, noise_level, (1, states_tensor.shape[1], states_tensor.shape[2])) * noise_mask
                    traj_noise = np.repeat(traj_noise, states_tensor.shape[0], axis=0)
                    perturbed_states = states_tensor + torch.tensor(traj_noise, device=self.device, dtype=torch.float64)
                    perturbed_rewards = -self.calc_traj_cost_batch(perturbed_states, emb_batch).detach().cpu().numpy()
                    perturbed_rewards = (perturbed_rewards - learned_rewards_min) / (learned_rewards_max - learned_rewards_min)
                    noisy_rewards.append(perturbed_rewards)
            # Compute variance of the learned rewards across perturbations.
            learned_reward_variance_list = np.var(noisy_rewards, axis=0)

            learned_reward_variance = np.mean(learned_reward_variance_list)
            reward_variance_list.append(learned_reward_variance)
            reward_variance_per_reward_densities[reward_density].append(learned_reward_variance)
        reward_variance_per_reward_densities = {k: np.mean(v) for k, v in reward_variance_per_reward_densities.items()}
        return np.mean(reward_variance_list), reward_variance_per_reward_densities

    def train(self, iterations, save_model=False, save_loss=False):
        losses = []
        # Retrieve batch size from params (default 10)
        self.params = self.params if hasattr(self, "params") else {}
        # batch_size = self.params.get("batch_size", 10)
        
        num_demos = self.demos.shape[0]
        num_train = self.all_trajs.shape[0]
        best_win_rate = 0.0
        
        p = tqdm.tqdm(range(iterations))
        for epoch in p:
            self.optimizer.zero_grad()
            epoch_loss = 0.0
            
            # Shuffle demo and training indices each epoch.
            demo_perm = torch.randperm(num_demos)
            train_perm = torch.randperm(num_train)
            
            # Iterate over demos in mini-batches.
            for i in range(0, num_demos, self.batch_size):
                demo_indices = demo_perm[i:i+self.batch_size]
                # Select mini-batch of demos.
                demo_batch = self.demos[demo_indices]
                # Compute language embeddings on the fly for demos.
                demo_instr_batch = [self.demo_language_instructions[idx] for idx in demo_indices.cpu().numpy()]
                demo_lang_emb_batch = self.lang_encoder(demo_instr_batch).to(torch.float64)
                
                # Sample a mini-batch from training trajectories.
                # To ensure a batch of the same size, we cycle or sample additional indices if necessary.
                train_indices = train_perm[i:i+self.batch_size]
                if len(train_indices) < self.batch_size:
                    additional = torch.randint(0, num_train, (self.batch_size - len(train_indices),))
                    train_indices = torch.cat([train_indices, additional])
                train_batch = self.all_trajs[train_indices]
                train_instr_batch = [self.train_language_instructions[idx] for idx in train_indices.cpu().numpy()]
                train_lang_emb_batch = self.lang_encoder(train_instr_batch).to(torch.float64)
                
                # Compute costs for the mini-batch of demo and training trajectories.
                cost_demos = self.calc_traj_cost_batch(demo_batch, demo_lang_emb_batch).unsqueeze(1)  # (B,)
                cost_train = self.calc_traj_cost_batch(train_batch, train_lang_emb_batch).unsqueeze(1)  # (B,)
                
                # Compute importance weights for this mini-batch.
                probs_demos = F.softmax(-cost_demos.squeeze(1).detach(), dim=0)
                probs_train = F.softmax(-cost_train.squeeze(1).detach(), dim=0)
                
                loss = torch.mean(cost_demos) + torch.log(
                    torch.mean(torch.exp(-cost_train)/probs_train) +
                    torch.mean(torch.exp(-cost_demos)/probs_demos)
                )
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item() * demo_batch.shape[0]
            
            epoch_loss /= num_demos
            p.set_description('Epoch %d loss: %.8f' % (epoch + 1, epoch_loss))
            losses.append(epoch_loss)

            # Log to wandb if enabled
            if self.wandb:
                cur_time = time.time()
                if epoch % 100 == 99:
                    results, avg_win_rate, gt_rewards_list, lr_rewards_list = self.evaluate(self.human_win_rates, return_rewards=True)
                    seen_theta_results, seen_theta_avg_win_rate, seen_theta_gt_rewards_list, seen_theta_lr_rewards_list = self.evaluate(self.seen_theta_human_win_rates, return_rewards=True)
                else:
                    results, avg_win_rate = self.evaluate(self.human_win_rates)
                    seen_theta_results, seen_theta_avg_win_rate = self.evaluate(self.seen_theta_human_win_rates)
                # reward variance
                learned_reward_variance_perturb_invalid, learned_reward_variance_per_reward_densities_invalid = self.evaluate_reward_variance(self.human_win_rates, noise_level=1.0)
                learned_reward_variance_perturb_valid, learned_reward_variance_per_reward_densities_valid = self.evaluate_reward_variance(self.human_win_rates, noise_level=1.0, perturb_valid=True)
                seen_theta_learned_reward_variance_perturb_invalid, seen_theta_learned_reward_variance_per_reward_densities_invalid = self.evaluate_reward_variance(self.seen_theta_human_win_rates, noise_level=1.0)
                seen_theta_learned_reward_variance_perturb_valid, seen_theta_learned_reward_variance_per_reward_densities_valid = self.evaluate_reward_variance(self.seen_theta_human_win_rates, noise_level=1.0, perturb_valid=True)
                # regret
                _, unseen_theta_avg_regret = self.evaluate_regret(self.human_win_rates)
                _, seen_theta_avg_regret = self.evaluate_regret(self.seen_theta_human_win_rates)


                wandb.log({
                    "train/epoch": epoch,
                    "train/loss": epoch_loss,
                    "train/maxent_loss": epoch_loss,
                    "eval/unseen_theta_avg_win_rate": avg_win_rate,
                    "eval/seen_theta_avg_win_rate": seen_theta_avg_win_rate,
                    "eval/unseen_theta_learned_reward_variance_perturb_invalid": learned_reward_variance_perturb_invalid,
                    "eval/unseen_theta_learned_reward_variance_perturb_valid": learned_reward_variance_perturb_valid,
                    "eval/seen_theta_learned_reward_variance_perturb_invalid": seen_theta_learned_reward_variance_perturb_invalid,
                    "eval/seen_theta_learned_reward_variance_perturb_valid": seen_theta_learned_reward_variance_perturb_valid,
                    "eval/unseen_theta_avg_regret": unseen_theta_avg_regret,
                    "eval/seen_theta_avg_regret": seen_theta_avg_regret,
                    
                }, step=epoch)
                
                # log reward variances for each density
                wandb.log({"eval/unseen_theta_learned_reward_variance_per_reward_{}_invalid".format(k): v for k, v in learned_reward_variance_per_reward_densities_invalid.items()}, step=epoch)
                wandb.log({"eval/unseen_theta_learned_reward_variance_per_reward_{}_valid".format(k): v for k, v in learned_reward_variance_per_reward_densities_valid.items()}, step=epoch)
                wandb.log({"eval/seen_theta_learned_reward_variance_per_reward_{}_invalid".format(k): v for k, v in seen_theta_learned_reward_variance_per_reward_densities_invalid.items()}, step=epoch)
                wandb.log({"eval/seen_theta_learned_reward_variance_per_reward_{}_valid".format(k): v for k, v in seen_theta_learned_reward_variance_per_reward_densities_valid.items()}, step=epoch)

                unseen_theta_result_dict, unseen_theta_result_dict_per_reward_density = count_avg_win_rate_per_num_valid_features(self.human_win_rates, results)
                seen_theta_result_dict, seen_theta_result_dict_per_reward_density = count_avg_win_rate_per_num_valid_features(self.seen_theta_human_win_rates, seen_theta_results)
                wandb.log({"eval/unseen_theta_valid_features{}_avg_win_rate".format(k): v for k, v in unseen_theta_result_dict.items()}, step=epoch)
                wandb.log({"eval/seen_theta_valid_features{}_avg_win_rate".format(k): v for k, v in seen_theta_result_dict.items()}, step=epoch)
                wandb.log({"eval/unseen_theta_reward_{}_avg_win_rate".format(k): v for k, v in unseen_theta_result_dict_per_reward_density.items()}, step=epoch)
                wandb.log({"eval/seen_theta_reward_{}_avg_win_rate".format(k): v for k, v in seen_theta_result_dict_per_reward_density.items()}, step=epoch)

                # also save results as wandb tabular
                table = wandb.Table(columns=["human_id", "theta", "win_rate", "epoch"])
                for i, info in enumerate(self.human_win_rates):
                    table.add_data(i, '_'.join(str(x) for x in info["theta"]), results[i], epoch)
                wandb.log({"eval/unseen_theta_win_rate_table": table}, step=epoch)

                table = wandb.Table(columns=["human_id", "theta", "win_rate", "epoch"])
                for i, info in enumerate(self.seen_theta_human_win_rates):
                    table.add_data(i, '_'.join(str(x) for x in info["theta"]), seen_theta_results[i], epoch)
                wandb.log({"eval/seen_theta_win_rate_table": table}, step=epoch)
                
                if epoch % 100 == 99:
                    unseen_theta_table = wandb.Table(columns=["human_id", "theta", "win_rate", "epoch", "gt_rewards", "lr_rewards"])
                    seen_theta_table = wandb.Table(columns=["human_id", "theta", "win_rate", "epoch", "gt_rewards", "lr_rewards"])

                    for i, info in enumerate(self.human_win_rates):
                        unseen_theta_table.add_data(i, '_'.join(str(x) for x in info["theta"]), results[i], epoch, gt_rewards_list[i], lr_rewards_list[i])
                    for i, info in enumerate(self.seen_theta_human_win_rates):
                        seen_theta_table.add_data(i, '_'.join(str(x) for x in info["theta"]), seen_theta_results[i], epoch, seen_theta_gt_rewards_list[i], seen_theta_lr_rewards_list[i])

                    wandb.log({"eval/unseen_theta_reward_table_full": unseen_theta_table}, step=epoch)
                    wandb.log({"eval/seen_theta_reward_table_full": seen_theta_table}, step=epoch)

                print("Epoch {}: Average win rate: {:.2f}%".format(epoch, avg_win_rate * 100))

                # Save the model if win rate improves.
                if save_model:
                    if avg_win_rate > best_win_rate:
                        best_win_rate = avg_win_rate
                        print("Epoch {}: Saving best model with win rate: {:.2f}%".format(epoch, best_win_rate * 100))
                        # just save full model including language encoder and cost_nn
                        torch.save({
                            'cost_nn': self.cost_nn.state_dict(),
                            'lang_encoder': self.lang_encoder.state_dict(),
                            'epoch': epoch,
                        }, self.best_ckpt)
                
                print("Epoch {}: Time taken: {:.2f} seconds".format(epoch, time.time() - cur_time))
        
        if save_model:
            torch.save({
                'cost_nn': self.cost_nn.state_dict(),
                'lang_encoder': self.lang_encoder.state_dict()
            }, self.last_ckpt)

        if save_loss:
            plt.plot(losses)
            plt.plot(maxent_losses)
            plt.plot(masked_losses)
            plt.legend(['Total loss', 'MaxEnt loss', 'Masked loss'])
            plt.savefig(os.path.join(save_dir, 'losses_it{}_lr{}.png'.format(iterations, self.lr)))
            plt.close()
        return losses
    
    def finetune(self, finetune_iterations=30, start_iteration=100, save_model=True):
        """
        Finetune the model on unseen test humans for a specified number of iterations.
        """
        if self.unseen_humans is None:
            print("No unseen humans provided for finetuning. Skipping finetuning.")
            return []
        
        print(f"Starting finetuning for {finetune_iterations} iterations on {len(self.unseen_humans)} unseen humans...")
        num_finetune_demos = self.finetune_demo_states.shape[0]
        finetune_losses = []
        best_win_rate = 0.0
        
        # Use a smaller learning rate for finetuning
        finetune_lr = self.lr * 0.1
        if self.use_state_encoder:
            finetune_optimizer = optim.Adam(list(self.cost_nn.parameters()) + list(self.state_encoder.parameters()) + list(self.lang_encoder.parameters()), lr=finetune_lr)
        else:
            finetune_optimizer = optim.Adam(list(self.cost_nn.parameters()) + list(self.lang_encoder.parameters()), lr=finetune_lr)
        
        p = tqdm.tqdm(range(finetune_iterations))
        for epoch in p:
            finetune_optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_maxent_loss = 0.0
            epoch_masked_loss = 0.0
            
            # Shuffle demo indices each epoch
            demo_perm = torch.randperm(num_finetune_demos)
            
            # Iterate over finetune demos in mini-batches
            for i in range(0, num_finetune_demos, self.batch_size):
                demo_indices = demo_perm[i:i+self.batch_size]
                if len(demo_indices) == 0:
                    continue
                
                # Select mini-batch of finetune demos
                demo_batch = self.finetune_demo_states[demo_indices]
                demo_instr_batch = [self.finetune_demo_language_instructions[idx] for idx in demo_indices.cpu().numpy()]
                demo_lang_emb_batch = self.lang_encoder(demo_instr_batch).to(torch.float64)
                
                # Sample training trajectories for contrast
                train_indices = torch.randint(0, self.all_trajs.shape[0], (len(demo_indices),))
                train_batch = self.all_trajs[train_indices]
                train_instr_batch = [self.train_language_instructions[idx] for idx in train_indices.cpu().numpy()]
                train_lang_emb_batch = self.lang_encoder(train_instr_batch).to(torch.float64)
                
                # Compute costs
                cost_demos = self.calc_traj_cost_batch(demo_batch, demo_lang_emb_batch).unsqueeze(1)
                cost_train = self.calc_traj_cost_batch(train_batch, train_lang_emb_batch).unsqueeze(1)
                
                # Compute importance weights
                probs_demos = F.softmax(-cost_demos.squeeze(1).detach(), dim=0)
                probs_train = F.softmax(-cost_train.squeeze(1).detach(), dim=0)
                
                loss = torch.mean(cost_demos) + torch.log(
                    torch.mean(torch.exp(-cost_train)/probs_train) +
                    torch.mean(torch.exp(-cost_demos)/probs_demos)
                )
                
                loss.backward()
                finetune_optimizer.step()
                finetune_optimizer.zero_grad()
                
                epoch_loss += loss.item() * demo_batch.shape[0]
            
            epoch_loss /= num_finetune_demos
            p.set_description('Epoch %d loss: %.8f' % (epoch + 1, epoch_loss))
            finetune_losses.append(epoch_loss)
            
            # Evaluate and log to wandb at each finetuning iteration
            if self.wandb:
                cur_time = time.time()
                results, avg_win_rate, gt_rewards_list, lr_rewards_list = self.evaluate(self.human_win_rates, return_rewards=True)
                seen_theta_results, seen_theta_avg_win_rate, seen_theta_gt_rewards_list, seen_theta_lr_rewards_list = self.evaluate(self.seen_theta_human_win_rates, return_rewards=True)
                # reward variance
                learned_reward_variance_perturb_invalid, learned_reward_variance_per_reward_densities_invalid = self.evaluate_reward_variance(self.human_win_rates, noise_level=1.0)
                learned_reward_variance_perturb_valid, learned_reward_variance_per_reward_densities_valid = self.evaluate_reward_variance(self.human_win_rates, noise_level=1.0, perturb_valid=True)
                seen_theta_learned_reward_variance_perturb_invalid, seen_theta_learned_reward_variance_per_reward_densities_invalid = self.evaluate_reward_variance(self.seen_theta_human_win_rates, noise_level=1.0)
                seen_theta_learned_reward_variance_perturb_valid, seen_theta_learned_reward_variance_per_reward_densities_valid = self.evaluate_reward_variance(self.seen_theta_human_win_rates, noise_level=1.0, perturb_valid=True)
                # regret
                _, unseen_theta_avg_regret = self.evaluate_regret(self.human_win_rates)
                _, seen_theta_avg_regret = self.evaluate_regret(self.seen_theta_human_win_rates)

                wandb.log({
                    "finetune/epoch": epoch,
                    "finetune/loss": epoch_loss,
                    "finetune/maxent_loss": epoch_loss,
                    "finetune/unseen_theta_avg_win_rate": avg_win_rate,
                    "finetune/seen_theta_avg_win_rate": seen_theta_avg_win_rate,
                    "finetune/unseen_theta_learned_reward_variance_perturb_invalid": learned_reward_variance_perturb_invalid,
                    "finetune/unseen_theta_learned_reward_variance_perturb_valid": learned_reward_variance_perturb_valid,
                    "finetune/seen_theta_learned_reward_variance_perturb_invalid": seen_theta_learned_reward_variance_perturb_invalid,
                    "finetune/seen_theta_learned_reward_variance_perturb_valid": seen_theta_learned_reward_variance_perturb_valid,
                    "finetune/unseen_theta_avg_regret": unseen_theta_avg_regret,
                    "finetune/seen_theta_avg_regret": seen_theta_avg_regret,
                }, step=start_iteration + epoch)
                
                wandb.log({"finetune/unseen_theta_learned_reward_variance_per_reward_{}_invalid".format(k): v for k, v in learned_reward_variance_per_reward_densities_invalid.items()}, step=start_iteration + epoch)
                wandb.log({"finetune/unseen_theta_learned_reward_variance_per_reward_{}_valid".format(k): v for k, v in learned_reward_variance_per_reward_densities_valid.items()}, step=start_iteration + epoch)

                unseen_theta_result_dict, unseen_theta_result_dict_per_reward_density = count_avg_win_rate_per_num_valid_features(self.human_win_rates, results)
                seen_theta_result_dict, seen_theta_result_dict_per_reward_density = count_avg_win_rate_per_num_valid_features(self.seen_theta_human_win_rates, seen_theta_results)
                wandb.log({"finetune/unseen_theta_valid_features{}_avg_win_rate".format(k): v for k, v in unseen_theta_result_dict.items()}, step=start_iteration + epoch)
                wandb.log({"finetune/seen_theta_valid_features{}_avg_win_rate".format(k): v for k, v in seen_theta_result_dict.items()}, step=start_iteration + epoch)
                wandb.log({"finetune/unseen_theta_reward_{}_avg_win_rate".format(k): v for k, v in unseen_theta_result_dict_per_reward_density.items()}, step=start_iteration + epoch)
                wandb.log({"finetune/seen_theta_reward_{}_avg_win_rate".format(k): v for k, v in seen_theta_result_dict_per_reward_density.items()}, step=start_iteration + epoch)

                # Log win rate tables
                # also save results as wandb tabular
                table = wandb.Table(columns=["human_id", "theta", "win_rate", "epoch", "gt_rewards", "lr_rewards"])
                for i, info in enumerate(self.human_win_rates):
                    table.add_data(i, '_'.join(str(x) for x in info["theta"]), results[i], start_iteration + epoch, gt_rewards_list[i], lr_rewards_list[i])
                wandb.log({"finetune/unseen_theta_win_rate_table": table}, step=start_iteration + epoch)

                table = wandb.Table(columns=["human_id", "theta", "win_rate", "epoch", "gt_rewards", "lr_rewards"])
                for i, info in enumerate(self.seen_theta_human_win_rates):
                    table.add_data(i, '_'.join(str(x) for x in info["theta"]), seen_theta_results[i], start_iteration + epoch, seen_theta_gt_rewards_list[i], seen_theta_lr_rewards_list[i])
                wandb.log({"finetune/seen_theta_win_rate_table": table}, step=start_iteration + epoch)

                print("FT Epoch {}: Average win rate: {:.2f}%".format(epoch, avg_win_rate * 100))

                # Save the model if win rate improves.
                if save_model:
                    if avg_win_rate > best_win_rate:
                        best_win_rate = avg_win_rate
                        print("Epoch {}: Saving best model with win rate: {:.2f}%".format(epoch, best_win_rate * 100))
                        # just save full model including language encoder and cost_nn
                        torch.save({
                            'cost_nn': self.cost_nn.state_dict(),
                            'lang_encoder': self.lang_encoder.state_dict(),
                            'epoch': epoch
                        }, self.finetune_best_ckpt)
                
                print("Epoch {}: Time taken: {:.2f} seconds".format(epoch, time.time() - cur_time))
                
        # save finetune_last_ckpt
        if save_model:
            torch.save({
                'cost_nn': self.cost_nn.state_dict(),
                'lang_encoder': self.lang_encoder.state_dict()
            }, self.finetune_last_ckpt)

        return finetune_losses