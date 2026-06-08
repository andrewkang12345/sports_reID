"""Split tracks at team-color flips. A track that contains both Argentina-light-blue
and France-navy frames is a tracker ID swap and should be split. This catches the
clearest within-track identity errors without needing trustworthy individual-player
discrimination.

Per-observation team classification: compare HSV hue of the shirt color to each team's
hex reference. A flip is detected when two consecutive observations vote different teams
with high confidence.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def hue(rgb):
    if rgb is None:
        return None, 0.0
    import cv2
    arr = np.clip(np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]
    return float(hsv[0]), float(hsv[1])


def hue_dist(h1: float, h2: float) -> float:
    diff = abs(h1 - h2)
    return min(diff, 180.0 - diff)


def parse_hex(s: str):
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def split_team_flips(input_dir: Path, output_dir: Path, sat_min: float = 30.0, flip_hue_margin: float = 20.0):
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in ("metadata.json", "visualization.mp4"):
        if (input_dir / f).exists():
            shutil.copy(input_dir / f, output_dir / f)
    result = json.loads((input_dir / "result.json").read_text())
    debug = json.loads((input_dir / "debug_tracks.json").read_text())
    meta = json.loads((input_dir / "metadata.json").read_text())

    # Build team reference hues
    team_hues = {}
    for team, c in (meta.get("team_colors") or {}).items():
        if isinstance(c, dict) and "shirt" in c:
            try:
                h, _ = hue(parse_hex(c["shirt"]))
                team_hues[team] = h
            except Exception:
                pass
    teams = list(team_hues.keys())
    print(f"Team hues: {team_hues}")

    def classify(obs):
        rgb = obs.get("team_color_rgb")
        if rgb is None: return None
        h, s = hue(rgb)
        if s < sat_min: return None
        # Find closest team by hue distance
        best = None; best_d = 999.0
        for team, th in team_hues.items():
            d = hue_dist(h, th)
            if d < best_d:
                best = team; best_d = d
        return best if best_d <= 35 else None  # close enough to a team

    new_debug = []
    next_id = 40000
    n_splits = 0
    summary_by_tid = {t["track_id"]: t for t in result["tracks"]}
    for tr in debug:
        if summary_by_tid.get(tr["track_id"], {}).get("role"):
            new_debug.append(tr); continue
        obs_list = sorted(tr["observations"], key=lambda o: int(o["frame_index"]))
        # Classify each observation
        cls_list = [classify(o) for o in obs_list]
        # Smooth: a single isolated different classification surrounded by another team is noise — ignore
        groups = [[]]
        cur_team = None
        for i, (obs, cls) in enumerate(zip(obs_list, cls_list)):
            if cls is None:
                groups[-1].append(obs); continue
            if cur_team is None:
                cur_team = cls
            elif cls != cur_team:
                # Check next observation too — only flip if the next ALSO agrees on the new team
                lookahead = next((cls_list[j] for j in range(i+1, min(i+4, len(cls_list))) if cls_list[j] is not None), None)
                if lookahead == cls:
                    groups.append([])
                    n_splits += 1
                    cur_team = cls
            groups[-1].append(obs)
        if len(groups) == 1:
            new_debug.append(tr); continue
        for gi, group in enumerate(groups):
            if not group: continue
            tid_out = tr["track_id"] if gi == 0 else str(next_id := next_id + 1)
            new_debug.append({
                "track_id": tid_out,
                "start_time": min(float(o["timestamp"]) for o in group),
                "end_time": max(float(o["timestamp"]) for o in group),
                "duration": max(float(o["timestamp"]) for o in group) - min(float(o["timestamp"]) for o in group),
                "is_player": True,
                "evidence": tr.get("evidence", {}),
                "observations": group,
            })
    print(f"Detected {n_splits} team-flip splits. Tracks {len(debug)} -> {len(new_debug)}")
    # Result tracks: keep existing summaries + add stub entries for new splits
    new_tids_in_debug = {nd["track_id"] for nd in new_debug}
    new_result = []
    summary_seen = set()
    for tr in result["tracks"]:
        if tr["track_id"] in new_tids_in_debug:
            new_result.append(tr); summary_seen.add(tr["track_id"])
    for nd in new_debug:
        if nd["track_id"] in summary_seen: continue
        # Stub entry for split child
        new_result.append({
            "track_id": nd["track_id"],
            "start_time": nd["start_time"],
            "end_time": nd["end_time"],
            "duration": nd["duration"],
            "is_player": True,
            "player_likelihood": 1.0,
            "resolved_player_id": None,
            "resolved_player": None,
            "resolved_confidence": 0.0,
            "best_jersey_guess": None,
            "jersey_vote_score": 0.0,
            "jersey_candidates": [],
            "jersey_peak_counts": {},
            "role": None,
            "team_probs": {},
            "split_from_team_flip": True,
        })
    out = dict(result); out["tracks"] = new_result
    (output_dir / "result.json").write_text(json.dumps(out, indent=2))
    (output_dir / "debug_tracks.json").write_text(json.dumps(new_debug, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    args = ap.parse_args()
    split_team_flips(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
