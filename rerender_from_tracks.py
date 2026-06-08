"""Re-derive jersey labels with the current aggregation logic and re-render the visualization
mp4 from a previous run's debug_tracks.json + source.mp4. Avoids re-running PARSeq.

Usage: python3 rerender_from_tracks.py outputs/clip_BRA_KOR_230503_v2 [outputs/clip_ARG_FRA_183303_v2 ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from soccer_identity.utils.schemas import load_yaml_config, load_metadata
from soccer_identity.utils.video_io import (
    create_video_writer,
    get_video_info,
    iter_video_frames,
    transcode_to_browser_mp4,
)


def best_jersey(observations, min_conf=0.40, peak_thresh=0.85, peak_min=4, longer_lock_min=3):
    """Return (jersey, score). Prefers a longer peak-locked candidate that contains
    the weighted-vote winner (so "14" wins over a "1" prefix when seen confidently).
    """
    votes: dict[str, float] = {}
    peak_counts: dict[str, int] = {}
    for obs in observations:
        jp = obs.get("jersey_probs") or {}
        jq = float(obs.get("jersey_quality", 0.0) or 0.0)
        for jersey, prob in jp.items():
            contribution = float(prob) * max(0.01, jq)
            votes[jersey] = votes.get(jersey, 0.0) + contribution
            if contribution >= peak_thresh:
                peak_counts[jersey] = peak_counts.get(jersey, 0) + 1
    if not votes:
        return None, 0.0
    total = sum(votes.values())
    if total <= 0:
        return None, 0.0
    weighted_best = max(votes, key=votes.get)
    share = votes[weighted_best] / total
    peaks = peak_counts.get(weighted_best, 0)
    longer = [j for j, c in peak_counts.items() if c >= longer_lock_min and len(j) > len(weighted_best) and weighted_best in j]
    if longer:
        chosen = max(longer, key=lambda j: (peak_counts[j], votes[j]))
        chosen_share = votes[chosen] / total
        chosen_score = votes[chosen] * chosen_share * (1.0 + min(1.0, peak_counts[chosen] / max(1, peak_min)))
        return chosen, chosen_score
    score = votes[weighted_best] * share * (1.0 + min(1.0, peaks / max(1, peak_min)))
    if peaks >= peak_min:
        return weighted_best, score
    if share >= min_conf:
        return weighted_best, score
    return None, score


def deduplicate(jersey_by_track: dict, scores: dict, team_by_track: dict) -> dict:
    by_key: dict[tuple[str | None, str], list[tuple[float, str]]] = {}
    for tid, jersey in jersey_by_track.items():
        if not jersey:
            continue
        team = team_by_track.get(tid)
        by_key.setdefault((team, jersey), []).append((scores.get(tid, 0.0), tid))
    deduped = dict(jersey_by_track)
    for items in by_key.values():
        items.sort(reverse=True)
        for _score, tid in items[1:]:
            deduped[tid] = None
    return deduped


def track_color(track_id: str):
    seed = int(track_id) if str(track_id).isdigit() else abs(hash(track_id))
    palette = [
        (58, 180, 255),
        (72, 214, 111),
        (255, 170, 68),
        (220, 96, 255),
        (92, 234, 225),
        (255, 96, 112),
    ]
    return palette[seed % len(palette)]


def rerender_clip(clip_dir: Path, config: dict, players_by_id: dict, metadata: dict | None = None):
    debug_tracks = json.loads((clip_dir / "debug_tracks.json").read_text())
    source = clip_dir / "source.mp4"
    output = clip_dir / "visualization.mp4"
    if not source.exists() or not debug_tracks:
        print(f"  skip {clip_dir} (missing source or empty tracks)")
        return

    viz = config.get("visualization", {})
    role = config.get("role_filter", {})
    min_player_likelihood = float(viz.get("min_player_likelihood", role.get("min_player_likelihood", 0.50)))
    min_track_duration = float(viz.get("min_track_duration_sec", 0.25))
    max_seconds = viz.get("max_seconds")
    h264_crf = int(viz.get("h264_crf", 22))
    min_jersey_conf = float(viz.get("min_jersey_display_confidence", 0.40))
    peak_thresh = float(viz.get("peak_lock_threshold", 0.85))
    peak_min = int(viz.get("peak_lock_min_count", 4))
    likely_threshold = float(viz.get("likely_threshold", 0.25))
    identity_threshold = float(config.get("assignment", {}).get("identity_confidence_threshold", 0.4))
    if metadata and metadata.get("identity_labels_available") is False:
        identity_threshold = 1.01
        likely_threshold = 1.01

    info = get_video_info(str(source))
    jersey_by_track: dict[str, str | None] = {}
    score_by_track: dict[str, float] = {}
    team_by_track: dict[str, str | None] = {}
    track_obs: dict[str, list[dict]] = {}
    for tr in debug_tracks:
        tid = tr["track_id"]
        if not tr.get("is_player"):
            jersey_by_track[tid] = None
            continue
        if float(tr.get("player_likelihood", 0.0)) < min_player_likelihood:
            jersey_by_track[tid] = None
            continue
        if float(tr.get("duration", 0.0)) < min_track_duration:
            jersey_by_track[tid] = None
            continue
        observations = tr.get("observations", [])
        jersey, score = best_jersey(observations, min_jersey_conf, peak_thresh, peak_min)
        jersey_by_track[tid] = jersey
        score_by_track[tid] = score
        # Team assignment: use evidence.team_probs if present in debug_tracks, else from observations team_color_rgb (skipped).
        ev = tr.get("evidence") or {}
        # team_probs aren't in debug_tracks evidence directly; look up identity_scores file in same dir
        team_by_track[tid] = None
        track_obs[tid] = observations

    # Try to fill team_by_track from debug_identity_scores.json.
    scores_path = clip_dir / "debug_identity_scores.json"
    if scores_path.exists():
        try:
            id_scores = json.loads(scores_path.read_text())
            for entry in id_scores:
                tid = entry.get("track_id")
                tp = (entry.get("evidence") or {}).get("team_probs") or {}
                if tid in team_by_track and tp:
                    team_by_track[tid] = max(tp, key=tp.get)
        except Exception:
            pass

    jersey_by_track = deduplicate(jersey_by_track, score_by_track, team_by_track)

    obs_by_frame: dict[int, list[tuple[dict, dict, str]]] = {}
    summary_updates: dict[str, str | None] = {}
    for tr in debug_tracks:
        tid = tr["track_id"]
        jersey = jersey_by_track.get(tid)
        summary_updates[tid] = jersey
        if jersey is None and (not tr.get("is_player") or float(tr.get("player_likelihood", 0.0)) < min_player_likelihood or float(tr.get("duration", 0.0)) < min_track_duration):
            continue  # this track is filtered from rendering
        observations = track_obs.get(tid)
        if observations is None:
            continue
        for obs in observations:
            obs_by_frame.setdefault(int(obs["frame_index"]), []).append((tr, obs, jersey))

    raw_path = output.with_name(f"{output.stem}.opencv_tmp{output.suffix}")
    writer = create_video_writer(str(raw_path), info.fps, (info.width, info.height))
    for frame_index, _ts, frame in iter_video_frames(str(source), max_seconds=max_seconds):
        overlay = frame.copy()
        for tr, obs, jersey in obs_by_frame.get(frame_index, []):
            color = track_color(tr["track_id"])
            x1, y1, x2, y2 = [int(v) for v in obs["bbox"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            resolved_id = tr.get("resolved_player_id")
            resolved_conf = float(tr.get("resolved_confidence", 0.0))
            player = players_by_id.get(resolved_id or "")
            if player is not None and resolved_conf >= identity_threshold:
                label = f"{player['player_name']} #{player.get('jersey_number') or '?'} {resolved_conf:.2f}"
            elif player is not None and resolved_conf >= likely_threshold:
                label = f"Likely {player['player_name']} #{player.get('jersey_number') or '?'} {resolved_conf:.2f}"
            elif jersey:
                label = f"Low conf ID #{jersey}"
            else:
                label = "Low conf ID"
            cv2.putText(overlay, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(overlay, f"ID {tr['track_id']}", (x1, min(overlay.shape[0] - 6, y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        writer.write(overlay)
    writer.release()
    if transcode_to_browser_mp4(raw_path, output, crf=h264_crf):
        raw_path.unlink(missing_ok=True)
    else:
        raw_path.replace(output)

    # Update result.json with new jersey assignments.
    result_path = clip_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        for tr in result.get("tracks", []):
            if tr["track_id"] in summary_updates:
                tr["best_jersey_guess"] = summary_updates[tr["track_id"]]
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"  rerendered {clip_dir.name} -> {output.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Per-clip output dirs containing debug_tracks.json + source.mp4 + metadata.json")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_yaml_config(root / args.config)
    for d in args.dirs:
        clip_dir = Path(d)
        meta_path = clip_dir / "metadata.json"
        meta = None
        if meta_path.exists():
            meta, players = load_metadata(str(meta_path))
            players_by_id = {p.player_id: p.to_dict() for p in players}
        else:
            players_by_id = {}
        rerender_clip(clip_dir, config, players_by_id, metadata=meta)


if __name__ == "__main__":
    main()
