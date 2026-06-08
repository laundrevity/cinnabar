"""Zero-pad an old checkpoint to the current GLOBAL_DIM so it can warm-start.

When GLOBAL_DIM grows (new global features appended LAST in encode_global), an old
checkpoint's first layers no longer match. This pads the global-input columns with
zeros — so the padded net behaves IDENTICALLY to the old one (the new features
contribute nothing until trained) — letting a warm-start isolate the new features'
effect instead of measuring "fresh vs warm-started".

    uv run python pad_checkpoint.py <in.pt> <out.pt> [hidden=128] [frames=1]

`frames>1` additionally frame-replicates: the global input grows to GLOBAL_DIM*frames with
the loaded weights placed in the LAST (current) frame and earlier frames zeroed — so the net
starts behaving identically (ignoring history) and a warm-started run learns to use it.
"""

import sys

import torch

from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM
from cinnabar.rl.net import ActionScorer


def pad(in_path: str, out_path: str, hidden: int = 128, frames: int = 1) -> None:
    sd = torch.load(in_path, map_location="cpu")
    pw = sd["policy_mlp.0.weight"]      # [hidden, old_global + ACTION_DIM]  (global cols first)
    vw = sd["value_mlp.0.weight"]       # [hidden, old_global]
    old_global = vw.shape[1]
    h = pw.shape[0]
    n_pad = GLOBAL_DIM - old_global
    if n_pad < 0:
        raise SystemExit(f"current GLOBAL_DIM {GLOBAL_DIM} < checkpoint's {old_global}")
    # 1) append zero columns so the global dim matches the current GLOBAL_DIM.
    if n_pad > 0:
        g_part, a_part = pw[:, :old_global], pw[:, old_global:]
        pw = torch.cat([g_part, torch.zeros(h, n_pad), a_part], dim=1)
        vw = torch.cat([vw, torch.zeros(h, n_pad)], dim=1)
    # 2) frame-replicate: weights into the LAST frame, earlier frames zeroed.
    if frames > 1:
        g_part, a_part = pw[:, :GLOBAL_DIM], pw[:, GLOBAL_DIM:]
        z = torch.zeros(h, (frames - 1) * GLOBAL_DIM)
        pw = torch.cat([z, g_part, a_part], dim=1)
        vw = torch.cat([z, vw], dim=1)
    sd["policy_mlp.0.weight"], sd["value_mlp.0.weight"] = pw, vw
    ActionScorer(GLOBAL_DIM * frames, ACTION_DIM, hidden).load_state_dict(sd)  # sanity: shapes line up
    torch.save(sd, out_path)
    print(f"wrote {out_path}: global {old_global} -> {GLOBAL_DIM} x {frames} frames = {GLOBAL_DIM * frames}")


if __name__ == "__main__":
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    frames = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    pad(sys.argv[1], sys.argv[2], hidden, frames)
