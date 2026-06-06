"""PGPolicy: a learnable Policy backed by the per-action scorer.

It plugs into the exact same `PolicyPlayer` path as RandomPolicy/MaxDamagePolicy.
In training mode it samples actions and records the trajectory (grouped by battle
tag); the trainer reads those out, attaches the win/loss reward, and updates the
net. In eval mode it greedily takes the highest-scoring action and records nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..encoding import featurize
from ..policy import Policy
from ..state import Action, BattleState
from .net import ActionScorer


@dataclass
class StepRecord:
    """One decision point, stored as plain features so the update can recompute
    log-probs/values with gradients (nothing from the rollout holds a graph)."""

    global_feats: list[float]
    action_feats: list[list[float]]
    chosen: int
    behavior_logp: float = 0.0  # log-prob of `chosen` under the rollout policy (PPO ratio)
    value: float = 0.0  # value estimate at rollout time (PPO baseline)


class PGPolicy(Policy):
    def __init__(self, net: ActionScorer, device: str = "cpu") -> None:
        self.net = net
        self.device = device
        self.training = True
        self.steps_by_battle: dict[str, list[StepRecord]] = {}

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def reset_buffer(self) -> None:
        self.steps_by_battle = {}

    def select_action(self, state: BattleState) -> Action:
        global_feats, action_feats = featurize(state)
        if not action_feats:
            raise ValueError("PGPolicy called with no available actions")

        g = torch.tensor(global_feats, dtype=torch.float32, device=self.device)
        a = torch.tensor(action_feats, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.net.score_actions(g, a)

        if self.training:
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=logits)
                sample = dist.sample()
                idx = int(sample.item())
                behavior_logp = float(dist.log_prob(sample))
                value = float(self.net.value(g))
            tag = state.battle_tag or "_unknown"
            self.steps_by_battle.setdefault(tag, []).append(
                StepRecord(global_feats, action_feats, idx, behavior_logp, value)
            )
        else:
            idx = int(torch.argmax(logits).item())

        return state.available_actions[idx]
