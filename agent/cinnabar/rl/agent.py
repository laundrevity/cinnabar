"""PGPolicy: a learnable Policy backed by the per-action scorer.

It plugs into the exact same `PolicyPlayer` path as RandomPolicy/MaxDamagePolicy.
In training mode it samples actions and records the trajectory (grouped by battle
tag); the trainer reads those out, attaches the win/loss reward, and updates the
net. In eval mode it greedily takes the highest-scoring action and records nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..encoding import TEAM_SIZE, featurize
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
    our_material: float = 0.0  # sum of our team HP fractions at decision time (dense reward)
    opp_material: float = 0.0  # 6 - damage dealt to revealed opp mons (dense reward)


class PGPolicy(Policy):
    def __init__(self, net: ActionScorer, device: str = "cpu") -> None:
        self.net = net
        self.device = device
        self.training = True
        self.record = True  # set False for a sampling opponent (self-play) that never trains
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
                if self.record:
                    behavior_logp = float(dist.log_prob(sample))
                    value = float(self.net.value(g))
            if self.record:
                our_material = sum(m.hp_fraction for m in state.team)
                revealed = state.opponent_team
                opp_material = sum(m.hp_fraction for m in revealed) + (TEAM_SIZE - len(revealed))
                tag = state.battle_tag or "_unknown"
                self.steps_by_battle.setdefault(tag, []).append(
                    StepRecord(global_feats, action_feats, idx, behavior_logp, value,
                               our_material, opp_material)
                )
        else:
            idx = int(torch.argmax(logits).item())

        return state.available_actions[idx]


class LeaguePolicy(Policy):
    """Opponent for league self-play. Each battle is played by a random snapshot
    drawn from the pool (chosen once per battle, sampled for diversity). It shares
    the pool list with the trainer, which appends new snapshots over time — so the
    learner must keep beating *all* its past selves, not just the latest. Never trains.
    """

    def __init__(self, nets: list[ActionScorer], device: str = "cpu") -> None:
        self.nets = nets  # shared with the training loop, which appends snapshots
        self.device = device
        self._choice: dict[str, int] = {}  # battle_tag -> snapshot index for that battle

    def select_action(self, state: BattleState) -> Action:
        global_feats, action_feats = featurize(state)
        if not action_feats:
            raise ValueError("LeaguePolicy called with no available actions")
        tag = state.battle_tag or "_unknown"
        if tag not in self._choice:
            self._choice[tag] = random.randrange(len(self.nets))
        net = self.nets[self._choice[tag]]

        g = torch.tensor(global_feats, dtype=torch.float32, device=self.device)
        a = torch.tensor(action_feats, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = net.score_actions(g, a)
        idx = int(torch.distributions.Categorical(logits=logits).sample().item())
        return state.available_actions[idx]
