import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class NNRewardPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_state: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_state)
        self.linear2 = nn.Linear(hidden_state, action_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def get_action(self, state: list, train: bool = False):
        # covnert to tensor
        state = torch.tensor(state, dtype=torch.float32)
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)

        # use stochastic sampling
        # Why?
        # we want to explore too not just exploit. if we only choose argmax, it might
        # only choose left or right since reward is always one until its failed.
        # choosing randomly gives us option to explore more.

        dist = Categorical(probs)
        if train:
            action = dist.sample()
        else:
            # for inference, use argmax
            action = torch.argmax(probs)

        log_prob = dist.log_prob(action)  # just the log of probability
        return action.item(), log_prob
