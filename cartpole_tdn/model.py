import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class CartpolePolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_state: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_state)
        self.linear2 = nn.Linear(hidden_state, action_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def get_action(self, state: list, train: bool = False):
        state = torch.tensor(state, dtype=torch.float32)
        logits = self.forward(state)

        probs = F.softmax(logits, dim=-1)

        dist = Categorical(probs)

        if train:
            action = dist.sample()
        else:
            action = torch.argmax(probs)

        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, return_dim: int, hidden_state: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_state)
        self.fc2 = nn.Linear(hidden_state, return_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_return(self, state):
        # covert to numpy . array
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float32)
        predicted_returns = self.forward(state)
        # flatten
        return predicted_returns.flatten()
