"""_stack_rows (the fast-path frame stacker) must match encoding.stack_global exactly."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from cinnabar.encoding import stack_global  # noqa: E402


def test_stack_rows_matches_stack_global():
    from train_engine import _stack_rows

    g, k, turns, b = 5, 4, 7, 3
    rng = np.random.default_rng(0)
    fast_hist = [[] for _ in range(b)]
    ref_hist = [[] for _ in range(b)]
    for t in range(turns):
        glob = rng.random((b, g)).astype(np.float32)
        stacked = _stack_rows(fast_hist, glob, k)
        for i in range(b):
            ref = stack_global(ref_hist[i], [float(x) for x in glob[i]], k)
            np.testing.assert_array_equal(stacked[i], np.asarray(ref, dtype=np.float32),
                                          err_msg=f"turn {t} battle {i}")
    assert all(len(h) == k - 1 for h in fast_hist)  # histories hold the previous k-1 frames
