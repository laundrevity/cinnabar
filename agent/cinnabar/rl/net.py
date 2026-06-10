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

    def _layer1(self, global_feats: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        """First layer, factored: Linear([g ++ a]) == W_g·g + W_a·a + bias, so the global
        half — the bulk of the input — is computed once per STATE instead of once per
        action row. Same parameters, same math (up to float summation order); profiling
        showed the fused layer-1 matmul dominating the PPO update."""
        l1 = self.policy_mlp[0]
        g_dim = global_feats.shape[-1]
        g_part = global_feats @ l1.weight[:, :g_dim].T + l1.bias       # (..., h)
        a_part = action_feats @ l1.weight[:, g_dim:].T                 # (..., N/K, h)
        return g_part.unsqueeze(-2) + a_part

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        x = torch.relu(hidden)
        x = torch.relu(self.policy_mlp[2](x))
        return self.policy_mlp[4](x).squeeze(-1)

    def score_actions(self, global_feats: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        """global_feats: (G,), action_feats: (N, A) -> logits: (N,)."""
        return self._head(self._layer1(global_feats, action_feats))  # (1,h)+(N,h) -> (N,)

    def score_actions_batch(self, global_feats: torch.Tensor, action_feats: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """Batched scorer for the PPO update.

        global_feats: (B, G), action_feats: (B, K, A), mask: (B, K) bool
        -> logits: (B, K), with -inf at padded (masked) action slots.
        """
        logits = self._head(self._layer1(global_feats, action_feats))
        return logits.masked_fill(~mask, float("-inf"))

    def value(self, global_feats: torch.Tensor) -> torch.Tensor:
        """global_feats: (G,) -> scalar value, or (B, G) -> (B,) batched."""
        return self.value_mlp(global_feats).squeeze(-1)


class ValueNet(nn.Module):
    """Standalone CALIBRATED win-probability head: global features -> P(win) in [0, 1].

    The ActionScorer's value head is trained as a PPO baseline on shaped returns — useful for
    *ranking* leaves, but uncalibrated as a probability (measured MSE ~0.48 vs win/loss, worse
    than always answering 0.5). This one is trained by supervised regression (BCE) on game
    outcomes from a deliberately diverse state distribution (train_value.py), so the search leaf
    can price tempo, incapacitation, and lost matchups instead of next-turn HP bars."""

    def __init__(self, global_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_feats: torch.Tensor) -> torch.Tensor:
        """Logits (for BCEWithLogits training)."""
        return self.mlp(global_feats).squeeze(-1)

    def value(self, global_feats: torch.Tensor) -> torch.Tensor:
        """P(win): (G,) -> scalar, or (B, G) -> (B,). The search-leaf interface."""
        return torch.sigmoid(self.forward(global_feats))


class HybridNet:
    """Search-facing shim: the POLICY net proposes (score_actions* feeds top-k gating), the
    calibrated ValueNet disposes (value() is the leaf). Duck-types ActionScorer for search.py —
    nothing in the search stack changes."""

    def __init__(self, policy_net: ActionScorer, value_net: ValueNet) -> None:
        self.policy_net = policy_net
        self.value_net = value_net

    def score_actions(self, g, a):
        return self.policy_net.score_actions(g, a)

    def score_actions_batch(self, g, a, mask):
        return self.policy_net.score_actions_batch(g, a, mask)

    def value(self, g):
        return self.value_net.value(g)
