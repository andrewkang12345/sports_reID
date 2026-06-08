"""Tracker + detection evaluator using the full-identity COCO ground truth.

Uses /mnt/data/sports_reID/groundTruth_AllTracking_ARG_FRA_183303 which has every
player in every sampled frame annotated — including "for tracker" instances where the
jersey number isn't visible. A correct tracker should identify these by maintaining
the SAME track ID as frames where the number IS visible (or via face/headshot/body).

Metrics:
  * detection_recall: fraction of GT bboxes with an IoU >= 0.5 predicted track
  * jersey_acc:       on the *number-visible* subset, fraction with correct jersey
  * identity_acc:     on ALL GT bboxes (including "for tracker"), fraction where the
                      predicted track maps to the correct player name. Mapping is by
                      majority vote of the player names assigned to each track via
                      its number-visible bboxes — i.e. once a track gets its number
                      read correctly even ONCE, all its other frames (including
                      no-number frames) should also be identified as that player.
  * id_switches:      number of cases where the same player appears in two
                      consecutive sampled frames but with different track IDs

Usage:
    python3 eval_tracking.py outputs/clip_ARG_FRA_183303_v19 [more runs ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

GT_DIR = Path("groundTruth_AllTracking_ARG_FRA_183303")
GT_JSON = GT_DIR / "_annotations.coco.json"


def _ascii_normalize(text: str) -> str:
    if not text:
        return text
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def parse_category(name: str) -> tuple[str | None, str | None, bool, bool]:
    """Return (jersey_number, player_name, is_visible_number, is_role).

    Categories take one of three forms:
      - 'referee'                          -> role
      - '10 Kylian Mbappe'                  -> visible number player
      - 'Goalkeeper 23 Emiliano Martinez'   -> goalkeeper (treat as visible)
      - 'for tracker 10 Kylian Mbappe'      -> tracker-only player (number occluded)
      - 'Player-Detection'                  -> skip (root category)
    """
    name = name.strip()
    if name == "Player-Detection":
        return None, None, False, False
    if name == "referee":
        return None, "Referee", False, True
    if name.startswith("for tracker "):
        rest = name[len("for tracker "):]
        m = re.match(r"(\d+)\s+(.+)", rest)
        if m:
            return m.group(1), m.group(2).strip(), False, False
        return None, rest.strip(), False, False
    if name.startswith("Goalkeeper "):
        rest = name[len("Goalkeeper "):]
        m = re.match(r"(\d+)\s+(.+)", rest)
        if m:
            return m.group(1), m.group(2).strip(), True, False
    m = re.match(r"(\d+)\s+(.+)", name)
    if m:
        return m.group(1), m.group(2).strip(), True, False
    return None, name, False, False


def iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    b_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = a_area + b_area - inter
    return inter / u if u > 0 else 0.0


def evaluate(run_dir: str) -> dict:
    rdir = Path(run_dir)
    res = json.loads((rdir / "result.json").read_text())
    debug = json.loads((rdir / "debug_tracks.json").read_text())
    coco = json.loads(GT_JSON.read_text())

    img_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}

    # Map COCO image index -> clip frame (mp4-NNNN <-> NNNN*30, capped at 899)
    def clip_frame_for(fname: str) -> int:
        m = re.search(r"mp4-(\d{4})", fname)
        return min(899, int(m.group(1)) * 30) if m else -1

    # Index predicted tracks by frame and by track_id
    track_meta = {t["track_id"]: t for t in res["tracks"]}
    obs_by_frame: dict[int, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for tr in debug:
        for obs in tr["observations"]:
            obs_by_frame[int(obs["frame_index"])].append((tr["track_id"], tuple(obs["bbox"])))

    # Index GT annotations by image_id and store (clip_frame, bbox, jersey, name, vis, role)
    gt_entries: list[tuple[int, tuple[float, float, float, float], str | None, str | None, bool, bool]] = []
    for ann in coco["annotations"]:
        cat = cat_by_id[ann["category_id"]]
        jersey, name, vis, role = parse_category(cat)
        if name is None:
            continue
        cf = clip_frame_for(img_by_id[ann["image_id"]])
        if cf < 0:
            continue
        x, y, w, h = ann["bbox"]
        gt_entries.append((cf, (x, y, x + w, y + h), jersey, _ascii_normalize(name), vis, role))

    # Stage 1: For each GT bbox, find best-IoU predicted track
    track_to_player_votes: dict[str, Counter] = defaultdict(Counter)
    gt_to_track: list[tuple[int, str | None, float]] = []  # parallel to gt_entries
    for cf, gt_box, _jersey, name, _vis, _role in gt_entries:
        best_iou = 0.0
        best_tid = None
        for tid, pbox in obs_by_frame.get(cf, []):
            i = iou_xyxy(gt_box, pbox)
            if i > best_iou:
                best_iou = i
                best_tid = tid
        gt_to_track.append((cf, best_tid, best_iou))
        if best_tid is not None and best_iou >= 0.5:
            track_to_player_votes[best_tid][name] += 1

    # Stage 2: Map each track to its majority-vote player name. Only count votes
    # from GT bboxes whose category was a *visible number* category (most reliable
    # identity signal). If no visible-number votes, fall back to all votes.
    track_visible_votes: dict[str, Counter] = defaultdict(Counter)
    for (cf, _gt_box, _jersey, name, vis, _role), (_cf2, tid, iou) in zip(gt_entries, gt_to_track):
        if tid is None or iou < 0.5 or not vis:
            continue
        track_visible_votes[tid][name] += 1
    track_to_player: dict[str, str] = {}
    for tid, votes in track_to_player_votes.items():
        if track_visible_votes.get(tid):
            track_to_player[tid] = track_visible_votes[tid].most_common(1)[0][0]
        else:
            track_to_player[tid] = votes.most_common(1)[0][0]

    # Stage 3: Per-bbox metrics
    n_total = 0
    n_matched = 0
    n_jersey_correct = 0
    n_jersey_visible = 0
    n_identity_correct = 0
    by_player = defaultdict(lambda: {"total": 0, "matched": 0, "id_correct": 0, "jersey_correct": 0})

    for (cf, _gt_box, jersey, name, vis, role), (_cf2, tid, iou) in zip(gt_entries, gt_to_track):
        if role:
            continue  # Skip referee for player metrics
        n_total += 1
        by_player[name]["total"] += 1
        if tid is None or iou < 0.5:
            continue
        n_matched += 1
        by_player[name]["matched"] += 1
        pred_name = track_to_player.get(tid)
        if pred_name == name:
            n_identity_correct += 1
            by_player[name]["id_correct"] += 1
        if vis and jersey is not None:
            n_jersey_visible += 1
            t = track_meta.get(tid, {})
            pred_jersey = t.get("best_jersey_guess")
            if pred_jersey is not None and str(pred_jersey) == str(jersey):
                n_jersey_correct += 1
                by_player[name]["jersey_correct"] += 1

    # Stage 4: ID switches — consecutive sampled frames with same player but different track IDs
    by_player_frame: dict[str, dict[int, str]] = defaultdict(dict)
    for (cf, _gt_box, _j, name, _vis, role), (_cf2, tid, iou) in zip(gt_entries, gt_to_track):
        if role or tid is None or iou < 0.5:
            continue
        by_player_frame[name][cf] = tid
    id_switches = 0
    for name, frame_to_tid in by_player_frame.items():
        frames = sorted(frame_to_tid.keys())
        for i in range(1, len(frames)):
            if frame_to_tid[frames[i]] != frame_to_tid[frames[i - 1]]:
                id_switches += 1

    return {
        "run": str(rdir),
        "n_total": n_total,
        "n_matched": n_matched,
        "n_jersey_visible": n_jersey_visible,
        "n_jersey_correct": n_jersey_correct,
        "n_identity_correct": n_identity_correct,
        "detection_recall_pct": 100.0 * n_matched / max(1, n_total),
        "jersey_acc_pct": 100.0 * n_jersey_correct / max(1, n_jersey_visible),
        "identity_acc_pct": 100.0 * n_identity_correct / max(1, n_total),
        "id_switches": id_switches,
        "track_to_player": track_to_player,
        "by_player": dict(by_player),
    }


def print_report(reports: list[dict]) -> None:
    print(f"\n{'Run':<55s} {'N':>4s} {'Det':>6s} {'Jersey':>7s} {'ID':>7s} {'IDsw':>5s}")
    print("-" * 90)
    for r in reports:
        run_name = r["run"].split("/")[-1]
        print(f"{run_name:<55s} {r['n_total']:>4d} "
              f"{r['detection_recall_pct']:>5.1f}% "
              f"{r['jersey_acc_pct']:>6.1f}% "
              f"{r['identity_acc_pct']:>6.1f}% "
              f"{r['id_switches']:>5d}")
    print("\nPer-player identity accuracy (best run by identity_acc):")
    best = max(reports, key=lambda r: r["identity_acc_pct"])
    print(f"  Best: {best['run'].split('/')[-1]}  id_acc {best['identity_acc_pct']:.1f}%")
    for name in sorted(best["by_player"].keys()):
        s = best["by_player"][name]
        if s["total"] == 0:
            continue
        print(f"  {name:<30s} id={s['id_correct']}/{s['total']}={100*s['id_correct']/s['total']:.0f}%   "
              f"jersey={s['jersey_correct']}/{s['matched']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    reports = []
    for d in args.dirs:
        try:
            r = evaluate(d)
            reports.append(r)
        except Exception as e:
            print(f"[{d}] ERROR: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
    print_report(reports)


if __name__ == "__main__":
    main()
