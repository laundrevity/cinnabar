"""The policy/value network: a per-action scorer.

Instead of a fixed action-index output, the network scores each *legal* action
from ``[global features ++ that action's features]`` and we softmax over those
scores. This handles a variable number of legal actions per turn and makes
illegal-action masking automatic (we simply never present illegal actions).

A separate value head estimates the state value from the global features, used as
the baseline in REINFORCE-with-baseline.
"""

from __future__ import annotations

import torch
from torch import nn


class ActionScorer(nn.Module):
    def __init__(self, global_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.policy_mlp = nn.Sequential(
            nn.Linear(global_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def score_actions(self, global_feats: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        """global_feats: (G,), action_feats: (N, A) -> logits: (N,)."""
        n = action_feats.shape[0]
        g = global_feats.unsqueeze(0).expand(n, -1)
        x = torch.cat([g, action_feats], dim=1)
        return self.policy_mlp(x).squeeze(-1)

    def value(self, global_feats: torch.Tensor) -> torch.Tensor:
        """global_feats: (G,) -> scalar state-value estimate."""
        return self.value_mlp(global_feats).squeeze(-1)
