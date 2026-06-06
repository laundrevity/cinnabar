"""Reinforcement-learning agent (Phase 2).

This subpackage depends on PyTorch, so it is NOT imported by ``cinnabar/__init__``
— the core package stays torch-free. Import from here explicitly:

    from cinnabar.rl.net import ActionScorer
    from cinnabar.rl.agent import PGPolicy

``cinnabar.rl.returns`` is itself torch-free (pure return/advantage math) so it
can be unit-tested without PyTorch installed.
"""
