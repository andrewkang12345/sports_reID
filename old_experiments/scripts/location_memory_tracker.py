"""Persistent location-memory tracker.

Insight: BoT-SORT tracks detections frame-by-frame without knowing identity. But once
a jersey number is *confidently* read at frame F, we KNOW that detection's identity —
and players move smoothly. We can use position memory to retroactively assign every
other detection of that player in nearby frames to the same identity, even when their
jersey isn't visible and BoT-SORT silently lost their ID.

Algorithm:
  1. Iterate ALL observations (across all original BoT-SORT tracks). Find those with
     a confident OCR read: max(prob * quality) >= 0.85 AND jersey is in the visible
     roster of the track's team_argmax. Each such observation is an "anchor" mapping
     (frame, x, y, appearance) -> a specific player_id.
  2. Group anchors by player_id. Build a per-player trajectory: list of (frame, cx, cy)
     and an averaged appearance fingerprint.
  3. For every other observation, compute (a) distance to each player's interpolated
     position at that frame, (b) appearance similarity to the player's avg embedding,
     (c) team consistency. Score = position-term + appearance-term + team-term.
  4. Per-frame Hungarian assignment (each obs <= one player, each player <= one obs).
     Threshold to avoid spurious matches.
  5. Output new debug_tracks.json where each player has ONE track_id consolidating all
     assigned observations.

This is the "remember who's who using location" approach.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def cosine(a, b):
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0


def bbox_center(bb):
    x1, y1, x2, y2 = bb
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def interp_position(anchors, frame_idx):
    """Linear interpolation/extrapolation of player position at a target frame from
    sorted (frame, x, y) anchors. Returns (x, y) or None if anchors is empty."""
    if not anchors:
        return None
    if len(anchors) == 1:
        return (anchors[0][1], anchors[0][2])
    # Binary-search-style: find anchors bracketing the target frame
    anchors = sorted(anchors)
    if frame_idx <= anchors[0][0]:
        # Extrapolate using first two
        f0, x0, y0 = anchors[0]; f1, x1, y1 = anchors[1]
        if f1 == f0: return (x0, y0)
        t = (frame_idx - f0) / (f1 - f0)
        return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    if frame_idx >= anchors[-1][0]:
        f0, x0, y0 = anchors[-2]; f1, x1, y1 = anchors[-1]
        if f1 == f0: return (x1, y1)
        t = (frame_idx - f0) / (f1 - f0)
        return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    # Interpolate within range
    for i in range(len(anchors) - 1):
        if anchors[i][0] <= frame_idx <= anchors[i + 1][0]:
            f0, x0, y0 = anchors[i]; f1, x1, y1 = anchors[i + 1]
            if f1 == f0: return (x0, y0)
            t = (frame_idx - f0) / (f1 - f0)
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    return None


def location_memory_assign(
    input_dir: Path,
    output_dir: Path,
    ocr_confirm_threshold: float = 0.85,
    max_position_distance: float = 150.0,
    min_appearance_sim: float = 0.40,
    score_threshold: float = 0.50,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in ("metadata.json", "visualization.mp4"):
        if (input_dir / f).exists():
            shutil.copy(input_dir / f, output_dir / f)
    result = json.loads((input_dir / "result.json").read_text())
    debug = json.loads((input_dir / "debug_tracks.json").read_text())
    meta = json.loads((input_dir / "metadata.json").read_text())

    # Roster lookups
    player_by_id = {}
    player_by_team_jersey = {}
    for team, roster in meta["rosters"].items():
        for p in roster:
            pid = f"{team}|{p['jersey_number']}|{p['player_name']}"
            player_by_id[pid] = p
            player_by_team_jersey[(team, str(p["jersey_number"]))] = pid
    visible = {team: {str(n) for n in nums} for team, nums in (meta.get("visible_jersey_numbers") or {}).items()}

    # Per-original-track team_argmax + role
    track_team = {}
    track_role = {}
    summary_by_tid = {t["track_id"]: t for t in result["tracks"]}
    for t in result["tracks"]:
        tp = t.get("team_probs") or {}
        track_team[t["track_id"]] = max(tp, key=tp.get) if tp else None
        track_role[t["track_id"]] = t.get("role")

    # === Step 1: Identify anchor observations ===
    # An anchor is an observation with a CONFIDENT OCR read for a jersey IN the visible
    # roster of the track's team_argmax. The anchor unambiguously names the player.
    anchors_per_player: dict[str, list[tuple[int, float, float, list[float]]]] = defaultdict(list)
    n_anchor_obs = 0
    for tr in debug:
        tid = tr["track_id"]
        team = track_team.get(tid)
        if not team or track_role.get(tid) in ("referee", "goalkeeper"):
            continue
        for obs in tr["observations"]:
            jp = obs.get("jersey_probs") or {}
            jq = float(obs.get("jersey_quality", 0.0) or 0.0)
            if jq <= 0: continue
            # Find best jersey for this team
            best_j = None; best_s = 0.0
            for j, p in jp.items():
                if visible and team in visible and str(j) not in visible[team]:
                    continue
                s = float(p) * jq
                if s > best_s:
                    best_s = s; best_j = j
            if best_j is None or best_s < ocr_confirm_threshold:
                continue
            pid = player_by_team_jersey.get((team, str(best_j)))
            if pid is None: continue
            cx, cy = bbox_center(obs["bbox"])
            anchors_per_player[pid].append((int(obs["frame_index"]), cx, cy, obs.get("appearance_embedding")))
            n_anchor_obs += 1
    print(f"Anchors: {sum(len(a) for a in anchors_per_player.values())} obs across {len(anchors_per_player)} players")
    for pid, a in anchors_per_player.items():
        print(f"  {player_by_id[pid]['player_name']}: {len(a)} anchor frame(s)")

    # === Step 2: Build per-player trajectory + appearance fingerprint ===
    trajectories: dict[str, list[tuple[int, float, float]]] = {}
    appearance_fp: dict[str, np.ndarray] = {}
    for pid, anchors in anchors_per_player.items():
        # Trajectory: keep ONE point per frame (median of multiple anchors if any)
        by_frame = defaultdict(list)
        embs = []
        for fr, x, y, emb in anchors:
            by_frame[fr].append((x, y))
            if emb is not None:
                embs.append(np.asarray(emb, dtype=np.float32))
        traj = sorted([(fr, np.median([p[0] for p in pts]), np.median([p[1] for p in pts]))
                       for fr, pts in by_frame.items()])
        trajectories[pid] = traj
        if embs:
            arr = np.stack(embs, axis=0)
            med = np.median(arr, axis=0)
            n = float(np.linalg.norm(med))
            appearance_fp[pid] = med / max(1e-8, n) if n > 1e-8 else med

    # === Step 3: For every non-anchor observation, score against each player ===
    # Group observations by frame
    obs_by_frame: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    obs_keys: dict[tuple[str, int], dict] = {}
    for tr in debug:
        tid = tr["track_id"]
        for obs in tr["observations"]:
            fr = int(obs["frame_index"])
            obs_by_frame[fr].append((tid, obs))
            obs_keys[(tid, fr)] = obs

    # === Step 4: Per-frame Hungarian assignment (CONSERVATIVE) ===
    # Only assign identity to observations whose ORIGINAL TRACK lacks a confident
    # resolved_player_id (or whose resolved player is NOT in the registry — i.e.,
    # likely a resolver hallucination). Never overwrite a BoT-SORT track that's
    # already firmly identified.
    track_resolved = {}
    for t in result["tracks"]:
        rpid = t.get("resolved_player_id")
        conf = float(t.get("resolved_confidence", 0.0) or 0.0)
        if rpid and conf >= 0.4:
            track_resolved[t["track_id"]] = rpid

    obs_to_player: dict[tuple[str, int], str] = {}
    n_assigned = 0
    for fr, obs_list in obs_by_frame.items():
        candidate_pids = list(trajectories.keys())
        if not candidate_pids: continue
        expected = {pid: interp_position(trajectories[pid], fr) for pid in candidate_pids}
        scored = []
        # Track which original-tracks are "frozen" (have a strong resolved id in registry)
        for oi, (tid, obs) in enumerate(obs_list):
            if track_role.get(tid) in ("referee", "goalkeeper"): continue
            # Skip if this track is already firmly identified to a registered player
            existing_pid = track_resolved.get(tid)
            if existing_pid and existing_pid in trajectories:
                # Lock this observation to its existing identity
                obs_to_player[(tid, fr)] = existing_pid
                n_assigned += 1
                continue
            team = track_team.get(tid)
            obs_cx, obs_cy = bbox_center(obs["bbox"])
            emb = obs.get("appearance_embedding")
            obs_emb = np.asarray(emb, dtype=np.float32) if emb else None
            for pid in candidate_pids:
                p_team = player_by_id[pid]["team_name"]
                if team and p_team and team != p_team: continue
                pos = expected.get(pid)
                if pos is None: continue
                px, py = pos
                dist = float(np.hypot(obs_cx - px, obs_cy - py))
                if dist > max_position_distance: continue
                pos_term = max(0.0, 1.0 - dist / max_position_distance)
                app_term = 0.0
                if obs_emb is not None and pid in appearance_fp:
                    sim = cosine(obs_emb, appearance_fp[pid])
                    if sim < min_appearance_sim: continue
                    app_term = (sim + 1.0) / 2.0
                else:
                    continue  # need appearance signal to be confident
                score = 0.5 * pos_term + 0.5 * app_term
                scored.append((score, oi, pid))
        # Greedy assignment for the not-yet-locked observations
        scored.sort(reverse=True)
        used_obs = {oi for oi, _ in [(oi, None) for oi, (tid, _) in enumerate(obs_list)
                                       if (tid, fr) in obs_to_player]}
        used_pid = set()
        # Lock claimed pids from already-resolved observations
        for (tid, fr_), pid in list(obs_to_player.items()):
            if fr_ == fr:
                used_pid.add(pid)
        for score, oi, pid in scored:
            if score < score_threshold: break
            if oi in used_obs or pid in used_pid: continue
            tid, obs = obs_list[oi]
            obs_to_player[(tid, fr)] = pid
            used_obs.add(oi); used_pid.add(pid)
            n_assigned += 1
    print(f"Per-frame assignments: {n_assigned} observations -> registered players")

    # === Step 5: Rebuild debug_tracks.json with per-player merged track_ids ===
    new_debug_groups: dict[str, list[dict]] = defaultdict(list)
    pid_to_new_tid = {}
    next_id = 50000
    for tr in debug:
        tid = tr["track_id"]
        if track_role.get(tid) in ("referee", "goalkeeper"):
            new_debug_groups[tid].extend(tr["observations"])
            continue
        for obs in tr["observations"]:
            fr = int(obs["frame_index"])
            pid = obs_to_player.get((tid, fr))
            if pid:
                if pid not in pid_to_new_tid:
                    next_id += 1
                    pid_to_new_tid[pid] = str(next_id)
                new_tid = pid_to_new_tid[pid]
            else:
                new_tid = tid
            new_debug_groups[new_tid].append(obs)

    # Emit final tracks
    final_debug = []
    for tid, obs_list in new_debug_groups.items():
        if not obs_list: continue
        obs_list = sorted(obs_list, key=lambda o: int(o["frame_index"]))
        final_debug.append({
            "track_id": tid,
            "start_time": min(float(o["timestamp"]) for o in obs_list),
            "end_time": max(float(o["timestamp"]) for o in obs_list),
            "duration": max(float(o["timestamp"]) for o in obs_list) - min(float(o["timestamp"]) for o in obs_list),
            "is_player": True,
            "evidence": {},
            "observations": obs_list,
        })

    # Update result.json — synthesize player-track entries for new track_ids, carry over rest
    final_result = []
    seen_tids = set()
    for tr in result["tracks"]:
        # Track was reassigned to some player(s); the player will own its own new tid below
        if tr["track_id"] in track_role and track_role[tr["track_id"]] in ("referee", "goalkeeper"):
            final_result.append(tr); seen_tids.add(tr["track_id"])
            continue
        # Check if this orig track's observations are ALL reassigned
        orig_tr = next((dt for dt in debug if dt["track_id"] == tr["track_id"]), None)
        if orig_tr:
            obs_remaining = sum(1 for obs in orig_tr["observations"]
                                if (tr["track_id"], int(obs["frame_index"])) not in obs_to_player)
            if obs_remaining > 0:
                final_result.append(tr); seen_tids.add(tr["track_id"])
    # New tracks per player
    for pid, new_tid in pid_to_new_tid.items():
        rp = player_by_id.get(pid, {})
        obs_for_player = new_debug_groups.get(new_tid, [])
        if not obs_for_player: continue
        final_result.append({
            "track_id": new_tid,
            "start_time": min(float(o["timestamp"]) for o in obs_for_player),
            "end_time": max(float(o["timestamp"]) for o in obs_for_player),
            "duration": max(float(o["timestamp"]) for o in obs_for_player) - min(float(o["timestamp"]) for o in obs_for_player),
            "is_player": True,
            "player_likelihood": 1.0,
            "resolved_player_id": pid,
            "resolved_player": rp,
            "resolved_confidence": 0.9,
            "best_jersey_guess": str(rp.get("jersey_number")) if rp.get("jersey_number") is not None else None,
            "jersey_vote_score": 1.0,
            "jersey_candidates": [],
            "jersey_peak_counts": {},
            "role": None,
            "team_probs": {rp.get("team_name", ""): 1.0} if rp else {},
            "location_memory_assigned": True,
        })

    out = dict(result); out["tracks"] = final_result
    (output_dir / "result.json").write_text(json.dumps(out, indent=2))
    (output_dir / "debug_tracks.json").write_text(json.dumps(final_debug, indent=2))
    print(f"Wrote {output_dir}: {len(final_result)} result tracks, {len(final_debug)} debug tracks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--ocr_threshold", type=float, default=0.85)
    ap.add_argument("--max_dist", type=float, default=150.0)
    ap.add_argument("--min_app_sim", type=float, default=0.40)
    ap.add_argument("--score_threshold", type=float, default=0.50)
    args = ap.parse_args()
    location_memory_assign(
        Path(args.input_dir), Path(args.output_dir),
        ocr_confirm_threshold=args.ocr_threshold,
        max_position_distance=args.max_dist,
        min_appearance_sim=args.min_app_sim,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
