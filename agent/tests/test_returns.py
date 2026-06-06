"""Tests for the torch-free return/advantage math."""

import pytest

from cinnabar.rl.returns import discounted_terminal_returns, standardize


def test_undiscounted_returns_are_flat():
    assert discounted_terminal_returns(3, 1.0, 1.0) == [1.0, 1.0, 1.0]


def test_discounted_returns_favor_later_steps():
    # gamma=0.5, length 3, terminal reward 1 -> [0.25, 0.5, 1.0]
    assert discounted_terminal_returns(3, 1.0, 0.5) == pytest.approx([0.25, 0.5, 1.0])


def test_loss_propagates_with_sign():
    assert discounted_terminal_returns(2, -1.0, 1.0) == [-1.0, -1.0]


def test_length_one_and_zero():
    assert discounted_terminal_returns(1, 1.0, 0.99) == [1.0]
    assert discounted_terminal_returns(0, 1.0, 0.99) == []


def test_standardize_zero_mean_unit_std():
    out = standardize([1.0, 2.0, 3.0, 4.0])
    assert sum(out) == pytest.approx(0.0, abs=1e-6)
    # population variance of standardized values ~ 1
    mean = sum(out) / len(out)
    var = sum((x - mean) ** 2 for x in out) / len(out)
    assert var == pytest.approx(1.0, abs=1e-6)


def test_standardize_constant_is_zeros():
    assert standardize([5.0, 5.0, 5.0]) == pytest.approx([0.0, 0.0, 0.0])


def test_standardize_empty():
    assert standardize([]) == []
