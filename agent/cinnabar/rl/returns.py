"""Return / advantage math — deliberately torch-free and unit-tested.

Credit assignment is where policy-gradient bugs love to hide, so this is kept as
plain Python with no dependencies.
"""

from __future__ import annotations


def discounted_terminal_returns(length: int, terminal_reward: float, gamma: float) -> list[float]:
    """Returns for an episode whose only reward arrives at the end (sparse).

    ``G_t = gamma**(length-1-t) * terminal_reward`` — later moves get more credit
    for the outcome, earlier ones are discounted.
    """
    if length <= 0:
        return []
    return [terminal_reward * (gamma ** (length - 1 - t)) for t in range(length)]


def standardize(values: list[float], eps: float = 1e-8) -> list[float]:
    """Zero-mean, unit-variance. Reduces gradient variance across a batch.

    A constant input maps to all zeros (no signal), which is the correct,
    harmless behaviour.
    """
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    return [(v - mean) / (std + eps) for v in values]
