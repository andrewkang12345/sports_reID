"""Conservative track-level merging via Player Identity Registry.

Per-observation reassignment was too noisy because OSNet on broadcast crops at this
resolution can't reliably distinguish individual players (kits + poses are similar).

Track-level approach:
  1. Build registry from CONFIDENT anchor tracks (visible roster + multi-peak OCR co-confirm).
  2. For each ORIGINAL track, compute its avg appearance fingerprint.
  3. If the track's appearance is overwhelmingly close to ONE registry player
     (cosine >= threshold AND margin >= 0.10 over the second-best AND team matches),
     merge that entire track into the registry player's track_id.
  4. Otherwise leave the track alone.

Effect: tracker eval sees fewer, longer tracks. Each merged track has the right player's
GT-vote majority. Conservative gate prevents propagating wrong identities.
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


def build_registry(tracks_summary, debug, metadata, min_resolved_conf: float = 0.4, min_peak_frames: int = 3):
    visible = {team: {str(n) for n in nums} for team, nums in (metadata.get("visible_jersey_numbers") or {}).items()}
    obs_by_tid = defaultdict(list)
    for tr in debug:
        for obs in tr["observations"]:
            e = obs.get("appearance_embedding")
            if not e: continue
            jp = obs.get("jersey_probs") or {}
            jq = float(obs.get("jersey_quality", 0.0) or 0.0)
            peak = max((float(p) * jq for p in jp.values()), default=0.0)
            obs_by_tid[tr["track_id"]].append((np.asarray(e, dtype=np.float32), peak))
    by_player = defaultdict(list)
    anchor = defaultdict(list)
    for t in tracks_summary:
        rpid = t.get("resolved_player_id")
        conf = float(t.get("resolved_confidence", 0) or 0)
        j = t.get("best_jersey_guess")
        rp = t.get("resolved_player") or {}
        if not (rpid and conf >= min_resolved_conf and j is not None and str(rp.get("jersey_number")) == str(j)):
            continue
        team = rp.get("team_name")
        if visible and team in visible and str(rp.get("jersey_number")) not in visible[team]:
            continue
        tid = t["track_id"]
        observations = obs_by_tid.get(tid, [])
        if sum(1 for _e, pk in observations if pk >= 0.85) < min_peak_frames:
            continue
        for emb, _pk in observations:
            by_player[rpid].append(emb)
        anchor[rpid].append(tid)
    registry = {}
    for pid, embs in by_player.items():
        if len(embs) < 5: continue
        arr = np.stack(embs, axis=0)
        med = np.median(arr, axis=0)
        n = float(np.linalg.norm(med))
        if n > 1e-8:
            registry[pid] = med / n
    return registry, anchor


def avg_emb(obs_list):
    embs = [np.asarray(o["appearance_embedding"], dtype=np.float32) for o in obs_list if o.get("appearance_embedding")]
    if len(embs) < 3: return None
    arr = np.stack(embs, axis=0)
    med = np.median(arr, axis=0)
    n = float(np.linalg.norm(med))
    return med / max(1e-8, n) if n > 1e-8 else None


def merge(input_dir: Path, output_dir: Path,
          threshold: float = 0.70, margin: float = 0.10, min_resolved_conf: float = 0.4):
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in ("metadata.json", "visualization.mp4"):
        if (input_dir / f).exists():
            shutil.copy(input_dir / f, output_dir / f)
    result = json.loads((input_dir / "result.json").read_text())
    debug = json.loads((input_dir / "debug_tracks.json").read_text())
    meta = json.loads((input_dir / "metadata.json").read_text())
    player_by_id = {f"{team}|{p['jersey_number']}|{p['player_name']}": p
                    for team, roster in meta["rosters"].items() for p in roster}

    registry, anchor = build_registry(result["tracks"], debug, meta, min_resolved_conf=min_resolved_conf)
    print(f"Registry: {len(registry)} players")
    for pid, anchors in anchor.items():
        if pid in registry:
            print(f"  {player_by_id.get(pid, {}).get('player_name', pid)}: {len(anchors)} anchor track(s)")

    summary_by_tid = {t["track_id"]: t for t in result["tracks"]}
    obs_by_tid = {tr["track_id"]: tr["observations"] for tr in debug}

    # Per-track: compute avg embedding + decide merge target
    track_to_pid: dict[str, str] = {}
    for tr in debug:
        tid = tr["track_id"]
        t = summary_by_tid.get(tid)
        if t is None or t.get("role"): continue
        # If track is itself an anchor, it's already a registry player
        for pid, anchors in anchor.items():
            if tid in anchors and pid in registry:
                track_to_pid[tid] = pid
                break
        if tid in track_to_pid: continue
        emb = avg_emb(tr["observations"])
        if emb is None: continue
        tp = t.get("team_probs") or {}
        team = max(tp, key=tp.get) if tp else None
        best = (None, -2.0); second = (None, -2.0)
        for pid, reg in registry.items():
            p_team = player_by_id.get(pid, {}).get("team_name")
            if team and p_team and team != p_team: continue
            sim = cosine(emb, reg)
            if sim > best[1]:
                second = best; best = (pid, sim)
            elif sim > second[1]:
                second = (pid, sim)
        if best[0] is None or best[1] < threshold: continue
        if best[1] - second[1] < margin: continue
        # Don't override a track that's already strongly resolved to a DIFFERENT player
        rpid_existing = t.get("resolved_player_id")
        existing_conf = float(t.get("resolved_confidence", 0) or 0)
        if rpid_existing and rpid_existing != best[0] and existing_conf >= 0.5:
            continue
        track_to_pid[tid] = best[0]
    print(f"Will merge {sum(1 for v in track_to_pid.values())} tracks into {len(set(track_to_pid.values()))} players")

    # Assign new track_id per player; keep unmatched tracks unchanged
    pid_to_new_tid = {}
    next_id = 20000
    new_debug = []
    seen_new_tids = set()
    for tr in debug:
        tid = tr["track_id"]
        target_pid = track_to_pid.get(tid)
        if target_pid:
            if target_pid not in pid_to_new_tid:
                next_id += 1
                pid_to_new_tid[target_pid] = str(next_id)
            new_tid = pid_to_new_tid[target_pid]
        else:
            new_tid = tid
        new_debug.append((new_tid, tr["observations"], tr.get("evidence", {})))
    # Group observations by new_tid (so multiple originals merge into one)
    grouped: dict[str, list] = defaultdict(list)
    grouped_ev: dict[str, dict] = {}
    for new_tid, obs_list, ev in new_debug:
        grouped[new_tid].extend(obs_list)
        if new_tid not in grouped_ev: grouped_ev[new_tid] = ev
    final_debug = []
    for new_tid, obs_list in grouped.items():
        if not obs_list: continue
        obs_list.sort(key=lambda o: int(o["frame_index"]))
        final_debug.append({
            "track_id": new_tid,
            "start_time": min(float(o["timestamp"]) for o in obs_list),
            "end_time": max(float(o["timestamp"]) for o in obs_list),
            "duration": max(float(o["timestamp"]) for o in obs_list) - min(float(o["timestamp"]) for o in obs_list),
            "is_player": True,
            "evidence": grouped_ev.get(new_tid, {}),
            "observations": obs_list,
        })
    # Rewrite result.json tracks
    final_result_tracks = []
    pid_seen = set()
    for tr in result["tracks"]:
        new_tid = pid_to_new_tid.get(track_to_pid.get(tr["track_id"])) if track_to_pid.get(tr["track_id"]) else tr["track_id"]
        target_pid = track_to_pid.get(tr["track_id"])
        if target_pid:
            if target_pid in pid_seen: continue
            pid_seen.add(target_pid)
            rp = player_by_id.get(target_pid, {})
            final_result_tracks.append({
                **tr,
                "track_id": new_tid,
                "resolved_player_id": target_pid,
                "resolved_player": rp,
                "resolved_confidence": max(float(tr.get("resolved_confidence", 0) or 0), 0.7),
                "best_jersey_guess": str(rp.get("jersey_number")) if rp.get("jersey_number") is not None else tr.get("best_jersey_guess"),
                "registry_merged": True,
            })
        else:
            final_result_tracks.append({**tr, "track_id": new_tid})
    out = dict(result); out["tracks"] = final_result_tracks
    (output_dir / "result.json").write_text(json.dumps(out, indent=2))
    (output_dir / "debug_tracks.json").write_text(json.dumps(final_debug, indent=2))
    print(f"Wrote {output_dir}: {len(final_result_tracks)} result tracks, {len(final_debug)} debug tracks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--min_resolved_conf", type=float, default=0.4)
    args = ap.parse_args()
    merge(Path(args.input_dir), Path(args.output_dir),
          threshold=args.threshold, margin=args.margin,
          min_resolved_conf=args.min_resolved_conf)


if __name__ == "__main__":
    main()
