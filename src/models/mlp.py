import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACT_FN = {"softplus": nn.Softplus(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hiddens, output_activation=None, dropout=0.0):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hiddens = hiddens

        model = []
        prev_h = np.prod(input_shape).item()
        for h in hiddens + [np.prod(output_shape).item()]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            if dropout != 0.0:
                model.append(nn.Dropout(dropout))
            prev_h = h

        model.pop()
        if dropout != 0.0:
            model.pop()

        if output_activation is not None:
            model.append(ACT_FN[output_activation])
        self.net = nn.Sequential(*model)

    def forward(self, x):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)
        return self.net(x.float())


class LSTMMLP(nn.Module):
    def __init__(self, input_shape, output_shape, hiddens):
        super().__init__()
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        self.lstm_layer = nn.LSTM(input_shape, hiddens[0])
        self.mlp = MLP(hiddens[0], output_shape, hiddens[1:])

    def forward(self, x):
        _, (hidden, _) = self.lstm_layer(x)
        hidden = F.leaky_relu(hidden[-1, :, :])
        return self.mlp(hidden)
