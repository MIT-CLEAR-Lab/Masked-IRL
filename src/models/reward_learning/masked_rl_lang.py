import os
import numpy as np
import torch
import tqdm
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import copy

from src.models.mlp import MLP
# Import the function that converts human theta to language instructions.
from src.utils.feature_utils import theta_to_language

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
    def __init__(self, feature_dim, cond_dim):
        """
        A FiLM block that modulates features based on a conditioning vector.
        """
        super(FiLMBlock, self).__init__()
        self.scale = nn.Linear(cond_dim, feature_dim)
        self.shift = nn.Linear(cond_dim, feature_dim)
    
    def forward(self, features, cond):
        gamma = self.scale(cond)
        beta = self.shift(cond)
        return gamma * features + beta

class FiLMRewardModel(nn.Module):
    def __init__(self, state_dim, cond_dim, hidden_sizes):
        """
        A reward model that conditions on state features and language embedding (condition)
        using FiLM layers.
        """
        super(FiLMRewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.film1 = FiLMBlock(hidden_sizes[0], cond_dim)
        layers = []
        in_dim = hidden_sizes[0]
        for h in hidden_sizes[1:]:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.fc_layers = nn.Sequential(*layers)
        self.film_final = FiLMBlock(in_dim, cond_dim)
        self.out = nn.Linear(in_dim, 1)
    
    def forward(self, state_features, cond):
        x = F.relu(self.fc1(state_features))
        x = self.film1(x, cond)
        x = self.fc_layers(x)
        x = self.film_final(x, cond)
        reward = self.out(x)
        return reward

##############################
# MaskedRL: Main Class       #
##############################
class MaskedRL:
    def __init__(self, params, featurizer, demos, all_trajs, 
                 train_features=None, demo_features=None, 
                 demo_thetas=None, train_thetas=None):
        """
        Reward model conditioned on language embeddings derived from human thetas.
        
        Args:
            params: configuration dictionary.
            featurizer: feature extractor for states.
            demos: raw demonstration trajectories.
            all_trajs: all trajectories for training IRL.
            train_features: features for training trajectories.
            demo_features: features for demonstration trajectories.
            demo_thetas: list/array of human theta vectors (one per demonstration).
            train_thetas: list/array of human theta vectors for training trajectories.
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.featurizer = featurizer
        if self.featurizer is not None and hasattr(self.featurizer, 'feat_scale_construct'):
            self.featurizer.feat_scale_construct(all_trajs)
        self.orig_demos = np.array(demos)
        self.demos = demo_features  # demonstration features
        self.orig_trajs = np.array(all_trajs)
        self.train_features = train_features
        self.all_trajs = train_features

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
        self.demo_language_instructions = theta_to_language(demo_thetas)
        self.train_language_instructions = theta_to_language(train_thetas)

        # Choose a language encoder option.
        # Options: "simple", "bert". Default to "simple".
        encoder_choice = params["language_encoder"] if "language_encoder" in params else "simple"
        vocab_size = params["language"].get("vocab_size", 10000)
        emb_dim = params["language"].get("emb_dim", 128)
        # We want the final language embedding to have dimension equal to theta_dim.
        theta_dim = self.demo_thetas.shape[1]
        if encoder_choice == "simple":
            self.lang_encoder = SimpleLanguageEncoder(vocab_size, emb_dim, theta_dim).to(self.device)
        elif encoder_choice == "bert":
            self.lang_encoder = BertLanguageEncoder(theta_dim).to(self.device)
        elif encoder_choice == "t5":
            self.lang_encoder = T5LanguageEncoder(theta_dim).to(self.device)
        else:
            raise ValueError(f"Unsupported language encoder: {encoder_choice}")

        # Precompute language embeddings for demos and training trajectories.
        # For demos:
        # self.demo_lang_emb = self.lang_encoder(self.tokenize_instructions(self.demo_language_instructions))
        # with torch.no_grad():
        #     self.demo_lang_emb = self.lang_encoder(self.demo_language_instructions)
        #     # For training, you may use similar language instructions if available.
        #     # self.train_lang_emb = self.lang_encoder(self.tokenize_instructions(self.train_language_instructions))
        #     self.train_lang_emb = self.lang_encoder(self.train_language_instructions)

        # For FiLM conditioning, we now use a FiLM reward network.
        feature_dim = self.demos.shape[1]
        hidden_sizes = params.get("hidden_sizes", [128, 128, 128])
        self.lr = params["lr"]
        self.batch_size = params.get("batch_size", 64)
        self.cost_nn = FiLMRewardModel(state_dim=feature_dim, cond_dim=theta_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.cost_nn = self.cost_nn.to(torch.float64)
        # self.optimizer = optim.Adam(self.cost_nn.parameters(), lr=params["lr"])
        # optimize the optimizer for the language encoder as well
        self.optimizer = optim.Adam(list(self.cost_nn.parameters()) + list(self.lang_encoder.parameters()), lr=self.lr)

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

    def calc_cost(self, trajs, trajs_features=None, lang_embs=None):
        if not torch.is_tensor(trajs):
            trajs = torch.as_tensor(trajs).to(self.device)
        if trajs_features is not None:
            if not torch.is_tensor(trajs_features):
                trajs_features = torch.as_tensor(trajs_features).to(self.device)
            features = trajs_features
        else:
            features = self.featurizer.featurize(trajs)
        if lang_embs is not None:
            if not torch.is_tensor(lang_embs):
                lang_embs = torch.as_tensor(lang_embs).float().to(self.device)
        else:
            raise ValueError("lang_embs must be provided.")
        cost = self.cost_nn(features, lang_embs)
        return cost.squeeze(1).detach().cpu().numpy()

    # def train(self, iterations, save_dir=None):
    #     losses = []
    #     maxent_losses = []
    #     masked_losses = []
    #     p = tqdm.tqdm(range(iterations))
        
    #     # Use precomputed demo language embeddings.
    #     # demos_concat = (self.demos, self.demo_lang_emb)  # For FiLM, pass separately.
    #     self.cost_nn = self.cost_nn.to(torch.float64)
    #     self.train_lang_emb = self.train_lang_emb.to(torch.float64)
    #     self.demos = self.demos.to(torch.float64)
    #     self.all_trajs = self.all_trajs.to(torch.float64)
    #     self.demo_lang_emb = self.demo_lang_emb.to(torch.float64)

    #     for epoch in p:
    #         self.optimizer.zero_grad()

    #         train_lang_emb = self.lang_encoder(self.train_language_instructions).to(torch.float64)
    #         # current_costs = self.cost_nn(self.all_trajs, self.train_lang_emb).squeeze(1)
    #         current_costs = self.cost_nn(self.all_trajs, train_lang_emb).squeeze(1)
    #         rand_indices = torch.multinomial(F.softmax(-current_costs, dim=0), len(self.demos), replacement=True)
    #         samples = self.all_trajs[rand_indices]
    #         # samples_lang = self.train_lang_emb[rand_indices]
    #         samples_lang = train_lang_emb[rand_indices]

    #         demo_lang_emb = self.lang_encoder(self.demo_language_instructions).to(torch.float64)
    #         # probs_demos = F.softmax(-self.cost_nn(self.demos, self.demo_lang_emb).squeeze(1).detach(), dim=0)
    #         probs_demos = F.softmax(-self.cost_nn(self.demos, demo_lang_emb).squeeze(1).detach(), dim=0)
    #         probs_samples = F.softmax(-self.cost_nn(samples, samples_lang).squeeze(1).detach(), dim=0)

    #         # cost_demos = self.cost_nn(self.demos, self.demo_lang_emb)
    #         cost_demos = self.cost_nn(self.demos, demo_lang_emb)
    #         cost_samples = self.cost_nn(samples, samples_lang)

    #         maxent_loss = torch.mean(cost_demos) + torch.log(
    #             torch.mean(torch.exp(-cost_samples)/probs_samples) + torch.mean(torch.exp(-cost_demos)/probs_demos)
    #         )

    #         num_repeat = 10
    #         repeated_features = self.demos.repeat(num_repeat, 1)
    #         # repeated_lang = self.demo_lang_emb.repeat(num_repeat, 1)
    #         repeated_lang = (demo_lang_emb).repeat(num_repeat, 1)
    #         random_noise = torch.randn_like(repeated_features) * 0.1  # adjust noise scale as needed
    #         perturbated_features = repeated_features + random_noise
    #         perturbated_cost = self.cost_nn(perturbated_features, repeated_lang)
    #         # masked_loss = torch.mean(torch.abs(self.cost_nn(self.demos, self.demo_lang_emb).repeat(num_repeat, 1) - perturbated_cost))
    #         masked_loss = torch.mean(torch.abs(self.cost_nn(self.demos, (demo_lang_emb)).repeat(num_repeat, 1) - perturbated_cost))

    #         loss = maxent_loss + masked_loss
    #         loss.backward()
    #         self.optimizer.step()

    #         p.set_description('Epoch %d loss: %.8f' % (epoch + 1, loss.item()))
    #         losses.append(loss.item())
    #         maxent_losses.append(maxent_loss.item())
    #         masked_losses.append(masked_loss.item())

    #     if save_dir is not None:
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         torch.save(self.cost_nn, os.path.join(save_dir, 'cost_network_it{}_lr{}.pt'.format(iterations, self.lr)))
    #         plt.plot(losses)
    #         plt.plot(maxent_losses)
    #         plt.plot(masked_losses)
    #         plt.legend(['Total loss', 'MaxEnt loss', 'Masked loss'])
    #         plt.savefig(os.path.join(save_dir, 'losses_it{}_lr{}.png'.format(iterations, self.lr)))
    #         plt.close()
    #     return losses

    def train(self, iterations, save_dir=None):
        losses = []
        maxent_losses = []
        masked_losses = []
        # Retrieve batch size from params (default 10)
        self.params = self.params if hasattr(self, "params") else {}
        # batch_size = self.params.get("batch_size", 10)
        
        num_demos = self.demos.shape[0]
        num_train = self.all_trajs.shape[0]
        
        p = tqdm.tqdm(range(iterations))
        for epoch in p:
            self.optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_maxent_loss = 0.0
            epoch_masked_loss = 0.0
            
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
                
                # Compute costs for the mini-batch.
                cost_demos = self.cost_nn(demo_batch, demo_lang_emb_batch)
                cost_train = self.cost_nn(train_batch, train_lang_emb_batch)
                
                # Compute importance weights for this mini-batch.
                probs_demos = F.softmax(-cost_demos.squeeze(1).detach(), dim=0)
                probs_train = F.softmax(-cost_train.squeeze(1).detach(), dim=0)
                
                maxent_loss = torch.mean(cost_demos) + torch.log(
                    torch.mean(torch.exp(-cost_train)/probs_train) +
                    torch.mean(torch.exp(-cost_demos)/probs_demos)
                )
                
                # Compute masked loss for demo batch.
                num_repeat = 10
                repeated_demo = demo_batch.repeat(num_repeat, 1)
                repeated_lang = demo_lang_emb_batch.repeat(num_repeat, 1)
                random_noise = torch.randn_like(repeated_demo) * 0.1
                # convert theta to mask by changing theta to absolute values and repeat
                mask = torch.abs(self.demo_thetas[demo_indices].repeat(num_repeat, 1))
                perturbed_demo = repeated_demo + random_noise * (1 - mask)
                perturbed_cost = self.cost_nn(perturbed_demo, repeated_lang)
                masked_loss = torch.mean(torch.abs(self.cost_nn(demo_batch, demo_lang_emb_batch).repeat(num_repeat, 1) - perturbed_cost))
                
                loss = maxent_loss + masked_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item() * demo_batch.shape[0]
                epoch_maxent_loss += maxent_loss.item() * demo_batch.shape[0]
                epoch_masked_loss += masked_loss.item() * demo_batch.shape[0]
            
            epoch_loss /= num_demos
            epoch_maxent_loss /= num_demos
            epoch_masked_loss /= num_demos
            p.set_description('Epoch %d loss: %.8f' % (epoch + 1, epoch_loss))
            losses.append(epoch_loss)
            maxent_losses.append(epoch_maxent_loss)
            masked_losses.append(epoch_masked_loss)
        
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.cost_nn, os.path.join(save_dir, 'cost_network_it{}_lr{}.pt'.format(iterations, self.lr)))
            plt.plot(losses)
            plt.plot(maxent_losses)
            plt.plot(masked_losses)
            plt.legend(['Total loss', 'MaxEnt loss', 'Masked loss'])
            plt.savefig(os.path.join(save_dir, 'losses_it{}_lr{}.png'.format(iterations, self.lr)))
            plt.close()
        return losses