import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PPOPolicy(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, return_dim: int = 1, hidden_state: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_state)
        self.actor_head = nn.Linear(hidden_state, action_dim)
        self.critic_head = nn.Linear(hidden_state, return_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        actor_logits = self.actor_head(x)
        critic_logits = self.critic_head(x)

        return actor_logits, critic_logits

    def act(
        self, state: list, train: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        actor_logits, critic_logits = self(state)  # __call__ calls forward so its equivalent to forward()

        # squeeze batch dim added by unsqueeze(0) above
        actor_logits = actor_logits.squeeze(0)
        value_estimate = critic_logits.squeeze(0).squeeze(-1)

        # actor
        dist = Categorical(logits=actor_logits)

        if train:
            action = dist.sample()
        else:
            action = torch.argmax(actor_logits)

        log_prob = dist.log_prob(action)

        entropy = dist.entropy()

        return action, value_estimate.squeeze(-1), log_prob, entropy
