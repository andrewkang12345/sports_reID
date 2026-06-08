"""Detect within-track ID switches by frame-to-frame spatial jumps + appearance drift,
and split the track at those points so the tracker eval sees identity-pure fragments.

Within a single BoT-SORT track, when two players cross paths, the tracker may swap
their IDs but keep the track alive. The bbox center jumps tens-hundreds of pixels
in one frame while passing through the same area. The appearance embedding also
drifts. Detecting these "switch frames" and splitting the track restores per-track
identity purity, which the tracker eval rewards via cleaner GT-vote majority.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def cosine(a, b):
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0


def split_tracks(input_dir: Path, output_dir: Path,
                 max_jump_px_per_frame: float = 80.0,
                 appearance_drop: float = 0.20):
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in ("metadata.json", "visualization.mp4"):
        if (input_dir / f).exists():
            shutil.copy(input_dir / f, output_dir / f)
    result = json.loads((input_dir / "result.json").read_text())
    debug = json.loads((input_dir / "debug_tracks.json").read_text())

    new_debug = []
    next_id = 30000
    n_splits = 0
    for tr in debug:
        obs_list = sorted(tr["observations"], key=lambda o: int(o["frame_index"]))
        # Compute per-frame jump and appearance drift
        cur_tid = tr["track_id"]
        cur_group = []
        groups = [cur_group]
        prev = None
        prev_emb = None
        for obs in obs_list:
            x1, y1, x2, y2 = obs["bbox"]
            cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
            fr = int(obs["frame_index"])
            emb = np.asarray(obs["appearance_embedding"], dtype=np.float32) if obs.get("appearance_embedding") else None
            split = False
            if prev is not None:
                px, py, pfr, _pemb = prev
                df = max(1, fr - pfr)
                dist = math.hypot(cx - px, cy - py)
                per_frame_jump = dist / df
                if per_frame_jump > max_jump_px_per_frame:
                    split = True
                # Appearance drop check
                if not split and emb is not None and prev_emb is not None:
                    sim = cosine(emb, prev_emb)
                    if sim < 0.5 - appearance_drop:  # appearance dropped significantly
                        split = True
            if split:
                cur_group = []
                groups.append(cur_group)
                n_splits += 1
            cur_group.append(obs)
            prev = (cx, cy, fr, emb)
            if emb is not None:
                prev_emb = emb
        # Emit each group as a separate track
        for gi, group in enumerate(groups):
            if not group: continue
            if gi == 0 and len(groups) == 1:
                tid_out = tr["track_id"]
            else:
                next_id += 1
                tid_out = str(next_id)
            new_debug.append({
                "track_id": tid_out,
                "start_time": min(float(o["timestamp"]) for o in group),
                "end_time": max(float(o["timestamp"]) for o in group),
                "duration": max(float(o["timestamp"]) for o in group) - min(float(o["timestamp"]) for o in group),
                "is_player": tr.get("is_player", True),
                "evidence": tr.get("evidence", {}),
                "observations": group,
            })
    # Result tracks: clone meta from original for the first group of each split
    # (lazy approach: leave existing result.json mostly as-is but add split children)
    # The eval's stage-2 majority vote will independently derive identity per new tid.
    summary_by_tid = {t["track_id"]: t for t in result["tracks"]}
    new_result = []
    seen_new_tids = set()
    for nd in new_debug:
        tid = nd["track_id"]
        if tid in seen_new_tids: continue
        seen_new_tids.add(tid)
        if tid in summary_by_tid:
            new_result.append(summary_by_tid[tid])
        else:
            # split child: synthesize a tracks-summary entry inheriting parent metadata
            parent_tid = None
            for orig in debug:
                if any(o["frame_index"] == nd["observations"][0]["frame_index"] and o["bbox"] == nd["observations"][0]["bbox"]
                       for o in orig["observations"]):
                    parent_tid = orig["track_id"]; break
            parent = summary_by_tid.get(parent_tid) if parent_tid else None
            entry = dict(parent) if parent else {"track_id": tid, "is_player": True}
            entry["track_id"] = tid
            entry["start_time"] = nd["start_time"]
            entry["end_time"] = nd["end_time"]
            entry["duration"] = nd["duration"]
            entry["split_from"] = parent_tid
            new_result.append(entry)
    print(f"Detected {n_splits} ID-switch splits. Tracks: {len(debug)} -> {len(new_debug)}")

    out = dict(result); out["tracks"] = new_result
    (output_dir / "result.json").write_text(json.dumps(out, indent=2))
    (output_dir / "debug_tracks.json").write_text(json.dumps(new_debug, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--max_jump", type=float, default=80.0)
    ap.add_argument("--appearance_drop", type=float, default=0.20)
    args = ap.parse_args()
    split_tracks(Path(args.input_dir), Path(args.output_dir),
                 max_jump_px_per_frame=args.max_jump,
                 appearance_drop=args.appearance_drop)


if __name__ == "__main__":
    main()
