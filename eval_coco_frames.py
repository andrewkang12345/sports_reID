"""End-to-end identity-accuracy evaluator against the per-frame COCO ground truth.

For each of the 31 labeled frames in groundTruth_ARG_FRA_183303/:
  1. Look up the matching clip frame (mp4-NNNN <-> clip_frame NNNN*30).
  2. From the run's debug_tracks.json, find each track's bbox at that frame.
  3. Greedily match predicted bboxes to GT bboxes by IoU >= 0.5.
  4. For each matched pair, render the predicted identity label
     (same logic as the renderer / eval_against_gt.py) and compare to GT.
  5. Report per-frame, per-jersey, and overall metrics.

This complements eval_against_gt.py (clip-level recall/precision) by measuring whether
the right ID actually appears on the right player AT THE RIGHT TIME.

Usage:
    python3 eval_coco_frames.py outputs/clip_ARG_FRA_183303_v28_mixsort_lowthresh [more runs ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


def _ascii_normalize(text: str) -> str:
    """Strip Unicode accents for comparison. COCO categories use ASCII names ('Kylian
    Mbappe') while our roster uses Unicode ('Kylian Mbappé'). Without this normalization
    the eval reports 0/22 for Mbappé even when the pipeline gets every frame right."""
    if not text:
        return text
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

GT_DIR = Path("groundTruth_ARG_FRA_183303")
GT_JSON = GT_DIR / "_annotations.coco.json"


def parse_jersey_from_category(name: str):
    if name in ("human", "referee", "Player-Detection"):
        return None, None, None
    m = re.match(r"(?:Goalkeeper\s+)?(\d+)\s+(.+)", name)
    if m:
        is_gk = name.startswith("Goalkeeper")
        return m.group(1), m.group(2).strip(), is_gk
    return None, None, None


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = a_area + b_area - inter
    return inter / u if u > 0 else 0.0


def render_label(t, player_by_id):
    role = t.get("role")
    if role == "referee":
        return "Referee", None, None
    if role == "goalkeeper":
        return "Goalkeeper", None, None
    j = t.get("best_jersey_guess")
    # Trust the resolver only when resolved_player_id is set (conflict resolver may have
    # nulled it). Stale resolved_player gets used only by the OCR-co-validated fallback.
    rp_active = t.get("resolved_player") if t.get("resolved_player_id") else None
    rp = rp_active
    conf = t.get("resolved_confidence", 0) if t.get("resolved_player_id") else 0
    ocr_confirms = rp and j and str(rp["jersey_number"]) == str(j)
    if rp and conf >= 0.4 and ocr_confirms:
        return f"{rp['player_name']} #{rp['jersey_number']}", rp["player_name"], str(rp["jersey_number"])
    if rp and conf >= 0.25 and ocr_confirms:
        return f"Likely {rp['player_name']} #{rp['jersey_number']}", rp["player_name"], str(rp["jersey_number"])
    if j:
        team_override = t.get("team_argmax_override")
        if team_override:
            team = team_override
        else:
            tp = t.get("team_probs") or {}
            team = max(tp, key=tp.get) if tp else None
        cands = [p for p in player_by_id.values() if str(p["jersey_number"]) == str(j)]
        if team:
            for p in cands:
                if p["team_name"] == team:
                    return f"{p['player_name']} #{j}", p["player_name"], str(j)
        if len(cands) == 1:
            return f"{cands[0]['player_name']} #{j}", cands[0]["player_name"], str(j)
    # v20a: high-confidence resolver fallback — when track lost dedup (jersey=None)
    # but resolver pinned a player via headshot/body/team with confidence >= 0.55,
    # trust the resolver and report that player.
    if rp and conf >= 0.55:
        return f"{rp['player_name']} #{rp.get('jersey_number') or '?'}", rp["player_name"], str(rp.get("jersey_number") or "")
    # v20b: resolver-supported candidate — even if the conflict resolver demoted this
    # track (resolved_player_id=None), the stale resolved_player dict may still name
    # a player. Trust it only when the track's top OCR candidate (pre-dedup) agrees
    # on the jersey AND scores above a threshold. This recovers Mbappé/Upamecano
    # fragments that have strong OCR support but lost dedup/conflict-resolution.
    rp_any = t.get("resolved_player")  # may be stale (resolved_player_id null) — gated by OCR
    if rp_any and rp_any.get("jersey_number") is not None:
        cands = t.get("jersey_candidates") or []
        if cands:
            top_j, top_s = cands[0]
            if str(top_j) == str(rp_any["jersey_number"]) and top_s >= 5.0:
                return f"{rp_any['player_name']} #{rp_any['jersey_number']}", rp_any["player_name"], str(rp_any["jersey_number"])
    return None, None, None


def evaluate(run_dir: str) -> dict:
    rdir = Path(run_dir)
    res = json.loads((rdir / "result.json").read_text())
    debug = json.loads((rdir / "debug_tracks.json").read_text())
    meta = json.loads((rdir / "metadata.json").read_text())
    coco = json.loads(GT_JSON.read_text())

    # Player table by id (team-aware)
    player_by_id: dict[str, dict] = {}
    for team, roster in meta["rosters"].items():
        for p in roster:
            pid = f"{team}|{p['jersey_number']}|{p['player_name']}"
            player_by_id[pid] = p

    # Index tracks by track_id, and observations by frame_index
    track_meta = {t["track_id"]: t for t in res["tracks"]}
    obs_by_frame: dict[int, list[tuple[str, tuple[float,float,float,float]]]] = defaultdict(list)
    for tr in debug:
        tid = tr["track_id"]
        for obs in tr["observations"]:
            obs_by_frame[int(obs["frame_index"])].append((tid, tuple(obs["bbox"])))

    # COCO frame_idx -> clip frame_idx (clip is 30fps, COCO is 1fps subsample)
    img_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    def coco_idx_of_image(fname: str) -> int:
        m = re.search(r"mp4-(\d{4})", fname)
        return int(m.group(1)) if m else -1
    def clip_frame_for(coco_idx: int) -> int:
        # 1 fps subsample -> clip frame = coco_idx * 30, capped at 899
        return min(899, coco_idx * 30)

    # Group annotations by image and filter to jersey-labeled categories only
    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        cat = cat_by_id[ann["category_id"]]
        gt_num, gt_name, is_gk = parse_jersey_from_category(cat)
        if gt_num is None:
            continue
        ann_by_image[ann["image_id"]].append({
            "bbox": ann["bbox"],
            "jersey": gt_num,
            "name": gt_name,
            "is_gk": is_gk,
        })

    # Per-frame matching: greedy IoU >= 0.5
    total_gt = 0
    correct_jersey = 0
    correct_name = 0
    matched = 0
    confusions = defaultdict(int)
    per_jersey_total = defaultdict(int)
    per_jersey_correct = defaultdict(int)

    for img_id, anns in ann_by_image.items():
        fname = img_by_id[img_id]
        cidx = coco_idx_of_image(fname)
        cframe = clip_frame_for(cidx)
        preds = obs_by_frame.get(cframe, [])

        # For each GT bbox, pick best-IoU predicted track and call its render_label
        used_preds = set()
        for ann in anns:
            x, y, w, h = ann["bbox"]
            gt_box = (x, y, x + w, y + h)
            total_gt += 1
            per_jersey_total[ann["jersey"]] += 1
            best_iou = 0.0
            best_pred = None
            for i, (tid, pbox) in enumerate(preds):
                if i in used_preds:
                    continue
                iou = iou_xyxy(gt_box, pbox)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = i
            if best_pred is None or best_iou < 0.50:
                confusions[(ann["jersey"], "<no-match>")] += 1
                continue
            used_preds.add(best_pred)
            matched += 1
            tid, _ = preds[best_pred]
            t = track_meta.get(tid)
            if t is None:
                confusions[(ann["jersey"], "<no-track-meta>")] += 1
                continue
            label, pred_name, pred_jersey = render_label(t, player_by_id)
            # GK has special handling
            if ann["is_gk"]:
                if label == "Goalkeeper":
                    correct_jersey += 1
                    correct_name += 1
                    per_jersey_correct[ann["jersey"]] += 1
                    confusions[(ann["jersey"], "Goalkeeper")] += 1
                else:
                    confusions[(ann["jersey"], label or "<low-conf>")] += 1
                continue
            jersey_ok = pred_jersey is not None and str(pred_jersey) == str(ann["jersey"])
            name_ok = pred_name is not None and _ascii_normalize(pred_name) == _ascii_normalize(ann["name"])
            if jersey_ok:
                correct_jersey += 1
                per_jersey_correct[ann["jersey"]] += 1
            if name_ok:
                correct_name += 1
            confusions[(ann["jersey"], pred_name or label or "<low-conf>")] += 1

    return {
        "run": str(rdir),
        "n_gt": total_gt,
        "n_matched": matched,
        "n_correct_jersey": correct_jersey,
        "n_correct_name": correct_name,
        "detection_recall_pct": 100.0 * matched / max(1, total_gt),
        "jersey_acc_pct": 100.0 * correct_jersey / max(1, total_gt),
        "name_acc_pct": 100.0 * correct_name / max(1, total_gt),
        "jersey_acc_when_matched_pct": 100.0 * correct_jersey / max(1, matched),
        "per_jersey": {j: (per_jersey_correct[j], per_jersey_total[j]) for j in per_jersey_total},
        "top_confusions": sorted(confusions.items(), key=lambda kv: -kv[1])[:20],
    }


def print_report(reports: list[dict]) -> None:
    print(f"\n{'Run':<55s} {'GT':>4s} {'Match':>6s} {'Jersey':>7s} {'Name':>6s}")
    print("-" * 85)
    for r in reports:
        run_name = r['run'].split('/')[-1]
        print(f"{run_name:<55s} {r['n_gt']:>4d} "
              f"{r['detection_recall_pct']:>5.1f}% "
              f"{r['jersey_acc_pct']:>6.1f}% "
              f"{r['name_acc_pct']:>5.1f}%")
    print("\nPer-jersey accuracy (best run):")
    best = max(reports, key=lambda r: r["jersey_acc_pct"])
    print(f"  Best: {best['run']} jersey-acc {best['jersey_acc_pct']:.1f}%")
    for j in sorted(best["per_jersey"].keys(), key=lambda x: int(x) if x.isdigit() else 999):
        c, n = best["per_jersey"][j]
        print(f"    #{j:<3}: {c}/{n} = {100*c/max(1,n):.0f}%")
    print("\nTop confusions (best run):")
    for (gt, pred), n in best["top_confusions"]:
        if str(gt) != str(pred):
            print(f"  GT #{gt} -> {pred}: {n}x")


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
    print_report(reports)


if __name__ == "__main__":
    main()
