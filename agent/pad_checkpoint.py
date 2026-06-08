"""Zero-pad an old checkpoint to the current GLOBAL_DIM so it can warm-start.

When GLOBAL_DIM grows (new global features appended LAST in encode_global), an old
checkpoint's first layers no longer match. This pads the global-input columns with
zeros — so the padded net behaves IDENTICALLY to the old one (the new features
contribute nothing until trained) — letting a warm-start isolate the new features'
effect instead of measuring "fresh vs warm-started".

    uv run python pad_checkpoint.py <in.pt> <out.pt> [hidden=128]
"""

import sys

import torch

from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM
from cinnabar.rl.net import ActionScorer


def pad(in_path: str, out_path: str, hidden: int = 128) -> None:
    sd = torch.load(in_path, map_location="cpu")
    pw = sd["policy_mlp.0.weight"]      # [hidden, old_global + ACTION_DIM]  (global cols first)
    vw = sd["value_mlp.0.weight"]       # [hidden, old_global]
    old_global = vw.shape[1]
    n_pad = GLOBAL_DIM - old_global
    if n_pad < 0:
        raise SystemExit(f"current GLOBAL_DIM {GLOBAL_DIM} < checkpoint's {old_global}")
    if n_pad == 0:
        torch.save(sd, out_path)
        print("no padding needed")
        return
    h = pw.shape[0]
    # policy input = [global | action]; insert the new global columns between them.
    g_part, a_part = pw[:, :old_global], pw[:, old_global:]
    sd["policy_mlp.0.weight"] = torch.cat([g_part, torch.zeros(h, n_pad), a_part], dim=1)
    # value input = [global]; append the new columns.
    sd["value_mlp.0.weight"] = torch.cat([vw, torch.zeros(h, n_pad)], dim=1)
    ActionScorer(GLOBAL_DIM, ACTION_DIM, hidden).load_state_dict(sd)  # sanity: shapes line up
    torch.save(sd, out_path)
    print(f"padded global {old_global} -> {GLOBAL_DIM} (+{n_pad} zero cols); wrote {out_path}")


if __name__ == "__main__":
    pad(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 128)
