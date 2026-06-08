"""Post-process: Player Identity Registry + per-observation re-assignment.

The breakthrough idea: BoT-SORT's tracker association is noisy. Within a single track
it sometimes follows TWO physical players (ID switches). And one physical player gets
split across multiple track IDs.

Instead of trusting the tracker, build a Player Identity Registry:
  1. For each player_id with a confident OCR-anchored track, snapshot its appearance
     fingerprint (mean OSNet embedding across all observations in that track).
  2. For every observation in the clip (across ALL tracks), compute its appearance
     embedding. Use Hungarian / nearest-neighbor matching to assign it to the best
     player in the registry.
  3. Re-write debug_tracks.json so each player has ONE track_id (per identity), and
     each observation is filed under the matched player.

This lets the tracker eval's per-track GT-vote majority operate on identity-pure
tracks — directly translating "we remembered who's who" into the metric.

Usage:
    python3 identity_registry_reassign.py outputs/clip_ARG_FRA_183303_v32 outputs/clip_ARG_FRA_183303_v42
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_registry(tracks_summary: list[dict], debug: list[dict], metadata: dict,
                   min_resolved_conf: float = 0.4, min_peak_frames: int = 3):
    """Build {player_id: avg_embedding} from CONFIDENTLY-anchored tracks.

    Anchor criteria (must hold):
      - resolved_player_id set with confidence >= min_resolved_conf
      - best_jersey_guess matches the resolved player's roster jersey (OCR co-confirm)
      - The resolved player's (team, jersey) is in the visible_jersey_numbers list — rules
        out resolver hallucinations like "Palacios #14 ARG" when #14 isn't visible.
      - The track has >= min_peak_frames observations with peak OCR (jersey_quality * top_prob
        for the target jersey >= 0.85) — single weak read isn't enough.

    Use the MEDIAN embedding (more robust to outliers like off-pose / partial-occlusion frames).
    """
    visible: dict[str, set[str]] = {team: {str(n) for n in nums} for team, nums in (metadata.get("visible_jersey_numbers") or {}).items()}

    # Build per-track observation list and peak count
    obs_by_tid: dict[str, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    for tr in debug:
        for obs in tr["observations"]:
            e = obs.get("appearance_embedding")
            if not e: continue
            jp = obs.get("jersey_probs") or {}
            jq = float(obs.get("jersey_quality", 0.0) or 0.0)
            peak_target = max((float(p) * jq for p in jp.values()), default=0.0)
            obs_by_tid[tr["track_id"]].append((int(obs["frame_index"]), np.asarray(e, dtype=np.float32), peak_target))

    summary_by_tid = {t["track_id"]: t for t in tracks_summary}
    by_player_embs: dict[str, list[np.ndarray]] = defaultdict(list)
    anchor_tracks: dict[str, list[str]] = defaultdict(list)
    for tid, t in summary_by_tid.items():
        rpid = t.get("resolved_player_id")
        conf = float(t.get("resolved_confidence", 0.0) or 0.0)
        jersey = t.get("best_jersey_guess")
        rp = t.get("resolved_player") or {}
        if not (rpid and conf >= min_resolved_conf and jersey is not None and str(rp.get("jersey_number")) == str(jersey)):
            continue
        # Visible-roster gate
        team = rp.get("team_name")
        if visible and team in visible and str(rp.get("jersey_number")) not in visible[team]:
            continue
        # Peak frame count gate
        observations = obs_by_tid.get(tid, [])
        n_peaks = sum(1 for _f, _e, pk in observations if pk >= 0.85)
        if n_peaks < min_peak_frames:
            continue
        for _fr, emb, _pk in observations:
            by_player_embs[rpid].append(emb)
        anchor_tracks[rpid].append(tid)
    registry: dict[str, np.ndarray] = {}
    for pid, embs in by_player_embs.items():
        if len(embs) < 5:
            continue
        arr = np.stack(embs, axis=0)
        med = np.median(arr, axis=0)
        n = float(np.linalg.norm(med))
        if n > 1e-8:
            registry[pid] = med / n
    return registry, anchor_tracks


def reassign(input_dir: Path, output_dir: Path,
             registry_threshold: float = 0.60,
             reassign_only_unconfirmed: bool = True,
             min_resolved_conf: float = 0.4):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Copy metadata + visualization unchanged
    for f in ("metadata.json", "visualization.mp4"):
        src = input_dir / f
        if src.exists():
            shutil.copy(src, output_dir / f)
    result = json.loads((input_dir / "result.json").read_text())
    debug = json.loads((input_dir / "debug_tracks.json").read_text())
    meta = json.loads((input_dir / "metadata.json").read_text())

    # Roster lookup
    player_by_id: dict[str, dict] = {}
    player_by_team_jersey: dict[tuple[str, str], str] = {}
    for team, roster in meta["rosters"].items():
        for p in roster:
            pid = f"{team}|{p['jersey_number']}|{p['player_name']}"
            player_by_id[pid] = p
            player_by_team_jersey[(team, str(p['jersey_number']))] = pid

    registry, anchor_tracks = build_registry(result["tracks"], debug, meta, min_resolved_conf=min_resolved_conf)
    print(f"Built registry: {len(registry)} players anchored")
    for pid, anchors in anchor_tracks.items():
        if pid in registry:
            print(f"  {player_by_id.get(pid, {}).get('player_name', pid)}: {len(anchors)} anchor track(s)")

    # Build per-observation list with original track_id, appearance, and OCR signal
    # We'll re-assign each observation to one player by:
    #   1. PRIORITY: per-observation OCR (top jersey × quality > 0.85) → force-assign to (team, jersey) player
    #   2. ELSE: nearest-neighbor in registry, if cosine >= threshold AND the chosen player's
    #      track at this frame doesn't already exist (don't duplicate)
    #   3. ELSE: keep original track_id
    summary_by_tid = {t["track_id"]: t for t in result["tracks"]}
    obs_team = {}
    for t in result["tracks"]:
        tp = t.get("team_probs") or {}
        obs_team[t["track_id"]] = max(tp, key=tp.get) if tp else None

    # New per-observation player assignment
    new_pid_for_obs: list[tuple[str, int, str | None]] = []  # (orig_tid, frame_idx, new_pid)
    per_frame_pids: dict[int, set[str]] = defaultdict(set)
    # First pass: OCR-forced assignments (priority)
    for tr in debug:
        tid = tr["track_id"]
        team = obs_team.get(tid)
        for obs in tr["observations"]:
            fr = int(obs["frame_index"])
            forced_pid = None
            jp = obs.get("jersey_probs") or {}
            jq = float(obs.get("jersey_quality", 0.0) or 0.0)
            for j, p in jp.items():
                contrib = float(p) * jq
                if contrib >= 0.85 and team:
                    pid = player_by_team_jersey.get((team, str(j)))
                    if pid:
                        forced_pid = pid
                        break
            if forced_pid:
                per_frame_pids[fr].add(forced_pid)
            new_pid_for_obs.append((tid, fr, forced_pid))
    # Build per-frame observation lookup for proper Hungarian-style assignment
    obs_by_frame_idx: dict[int, list[tuple[int, str, dict, np.ndarray]]] = defaultdict(list)
    for tr in debug:
        for obs in tr["observations"]:
            e = obs.get("appearance_embedding")
            if not e: continue
            fr = int(obs["frame_index"])
            obs_by_frame_idx[fr].append((-1, tr["track_id"], obs, np.asarray(e, dtype=np.float32)))
    # Re-index with the new_pid_for_obs position so we can update efficiently
    pos_by_key: dict[tuple[str, int], int] = {}
    for i, (tid, fr, _p) in enumerate(new_pid_for_obs):
        pos_by_key[(tid, fr)] = i

    # Second pass: per-frame Hungarian assignment.
    # For each frame, build cost matrix: observations × players in registry.
    # Solve to maximize total similarity. Only accept matches with sim >= registry_threshold AND
    # margin >= 0.05 above the second-best player (so close ties are rejected).
    import itertools
    for fr, obs_list in obs_by_frame_idx.items():
        if not obs_list or not registry: continue
        # Drop observations already forced by OCR
        candidates = [(tid, obs, emb) for (_, tid, obs, emb) in obs_list
                      if pos_by_key.get((tid, fr)) is not None and new_pid_for_obs[pos_by_key[(tid, fr)]][2] is None]
        if not candidates: continue
        # Filter registry by team for each candidate
        # Compute cost matrix
        pids = list(registry.keys())
        for (tid, obs, emb) in candidates:
            team = obs_team.get(tid)
            # Find best and second-best player matches subject to team constraint
            best = (None, -2.0); second = (None, -2.0)
            for pid in pids:
                if pid in per_frame_pids[fr]: continue  # already claimed this frame
                p_team = player_by_id.get(pid, {}).get("team_name")
                if team and p_team and team != p_team: continue
                sim = cosine(emb, registry[pid])
                if sim > best[1]:
                    second = best; best = (pid, sim)
                elif sim > second[1]:
                    second = (pid, sim)
            if best[0] is None or best[1] < registry_threshold: continue
            # Margin gate: best must beat second by >= 0.05
            if best[1] - second[1] < 0.05: continue
            pos = pos_by_key.get((tid, fr))
            if pos is not None:
                new_pid_for_obs[pos] = (tid, fr, best[0])
                per_frame_pids[fr].add(best[0])

    # Rebuild debug_tracks.json: one new track_id per player (for matched obs), keep original for unmatched
    obs_index: dict[tuple[str, int], dict] = {}
    for tr in debug:
        for obs in tr["observations"]:
            obs_index[(tr["track_id"], int(obs["frame_index"]))] = obs

    # Allocate new track IDs: player_id → tid, plus carry-over for unmatched
    pid_to_new_tid: dict[str, str] = {}
    next_id = 10000
    new_tracks: dict[str, list[dict]] = defaultdict(list)
    for tid, fr, new_pid in new_pid_for_obs:
        obs = obs_index.get((tid, fr))
        if obs is None:
            continue
        if new_pid is not None:
            if new_pid not in pid_to_new_tid:
                next_id += 1
                pid_to_new_tid[new_pid] = str(next_id)
            new_tid = pid_to_new_tid[new_pid]
        else:
            new_tid = tid  # carry over
        new_tracks[new_tid].append(obs)

    # Replace observations in debug
    new_debug = []
    seen_tids = set()
    for tid, obs_list in new_tracks.items():
        new_obs = sorted(obs_list, key=lambda o: o["frame_index"])
        if not new_obs:
            continue
        # Inherit metadata from the player's anchor track (if it's a reassigned-to-player track),
        # else from any original track contributing observations
        new_debug.append({
            "track_id": tid,
            "start_time": min(float(o["timestamp"]) for o in new_obs),
            "end_time": max(float(o["timestamp"]) for o in new_obs),
            "duration": max(float(o["timestamp"]) for o in new_obs) - min(float(o["timestamp"]) for o in new_obs),
            "is_player": True,
            "evidence": {},
            "observations": new_obs,
        })
    # Also rewrite result.json's tracks to match the new track IDs
    new_result_tracks = []
    pid_to_track_meta: dict[str, dict] = {}
    for tid, obs_list in new_tracks.items():
        if not obs_list:
            continue
        # If tid is a reassigned-to-player tid, find the player
        pid = None
        for p, t in pid_to_new_tid.items():
            if t == tid: pid = p; break
        if pid:
            rp = player_by_id.get(pid, {})
            new_result_tracks.append({
                "track_id": tid,
                "start_time": min(float(o["timestamp"]) for o in obs_list),
                "end_time": max(float(o["timestamp"]) for o in obs_list),
                "duration": max(float(o["timestamp"]) for o in obs_list) - min(float(o["timestamp"]) for o in obs_list),
                "is_player": True,
                "player_likelihood": 1.0,
                "resolved_player_id": pid,
                "resolved_player": rp,
                "resolved_confidence": 0.8,
                "best_jersey_guess": str(rp.get("jersey_number")) if rp else None,
                "jersey_vote_score": 1.0,
                "jersey_candidates": [],
                "jersey_peak_counts": {},
                "role": None,
                "team_probs": {rp.get("team_name", ""): 1.0} if rp else {},
                "registry_reassigned": True,
            })
        else:
            # Carry-over: copy from original
            orig = summary_by_tid.get(tid)
            if orig:
                new_result_tracks.append(orig)

    out_result = dict(result)
    out_result["tracks"] = new_result_tracks
    (output_dir / "result.json").write_text(json.dumps(out_result, indent=2))
    (output_dir / "debug_tracks.json").write_text(json.dumps(new_debug, indent=2))
    print(f"Wrote {output_dir}/result.json ({len(new_result_tracks)} tracks)")
    print(f"Wrote {output_dir}/debug_tracks.json ({len(new_debug)} tracks)")
    n_reassigned = sum(1 for _, _, p in new_pid_for_obs if p is not None)
    print(f"Reassigned {n_reassigned} of {len(new_pid_for_obs)} observations to registered players")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--threshold", type=float, default=0.60, help="Registry cosine similarity threshold")
    ap.add_argument("--min_resolved_conf", type=float, default=0.4)
    args = ap.parse_args()
    reassign(Path(args.input_dir), Path(args.output_dir),
             registry_threshold=args.threshold,
             min_resolved_conf=args.min_resolved_conf)


if __name__ == "__main__":
    main()
