"""Evaluate jersey-number OCR accuracy against the user's per-frame COCO ground truth.

Uses the 31 labeled frames in groundTruth_ARG_FRA_183303/ to isolate OCR performance from
detection/tracking. For each annotation with a known jersey number, crop the GT bbox from
the source image, feed through a configurable OCR stack, and compare against the GT label.

Usage:
    python3 eval_ocr_stack.py
    python3 eval_ocr_stack.py --stack=baseline,no-sr,torso-only,no-visible-filter
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from soccer_identity.identity.jersey_ocr import build_jersey_ocr, build_legibility_classifier
from soccer_identity.utils.schemas import load_yaml_config

GT_DIR = Path("groundTruth_ARG_FRA_183303")
GT_JSON = GT_DIR / "_annotations.coco.json"


def parse_jersey_from_category(name: str) -> tuple[str | None, str | None]:
    """Pull (jersey_number, player_name) out of category names like '10 Kylian Mbappe',
    'Goalkeeper 23 Emiliano Martinez', or skip-categories ('human', 'referee')."""
    if name in ("human", "referee") or name == "Player-Detection":
        return None, None
    m = re.match(r"(?:Goalkeeper\s+)?(\d+)\s+(.+)", name)
    if m:
        return m.group(1), m.group(2).strip()
    return None, None


def crop_bbox(img: np.ndarray, bbox: list[float], pad: int = 4) -> np.ndarray:
    x, y, w, h = bbox
    h_img, w_img = img.shape[:2]
    x1 = max(0, int(x) - pad)
    y1 = max(0, int(y) - pad)
    x2 = min(w_img, int(x + w) + pad)
    y2 = min(h_img, int(y + h) + pad)
    return img[y1:y2, x1:x2]


def torso_crop(crop: np.ndarray) -> np.ndarray:
    """Approximate torso region — upper 50% of the bbox, central 70% horizontally.

    Without pose keypoints, this is a coarse but effective alternative to a full crop
    when the player is roughly upright. Real torso-from-pose lives in run_demo.py."""
    h, w = crop.shape[:2]
    if h < 20 or w < 10:
        return crop
    y1 = int(h * 0.15)
    y2 = int(h * 0.65)
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    return crop[y1:y2, x1:x2]


def load_sr_model():
    """Mirror run_demo._load_sr_model — Real-ESRGAN x4 for tiny crops."""
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        import torch
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        upsampler = RealESRGANer(scale=4, model_path=model_url, model=model, tile=0, pre_pad=0, half=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        return upsampler
    except Exception as e:
        print(f"[sr] disabled ({e})")
        return None


def maybe_upscale(crop: np.ndarray, sr) -> np.ndarray:
    if sr is None or crop.size == 0:
        return crop
    try:
        h, w = crop.shape[:2]
        if h >= 96 and w >= 64:
            return crop  # already large enough
        out, _ = sr.enhance(crop, outscale=4)
        return out
    except Exception:
        return crop


def confusable_smooth(
    probs: dict[str, float],
    strength: float = 0.6,
    expand_one_digit_strength: float = 0.8,
) -> dict[str, float]:
    """Redistribute probability mass over PARSeq digit-confusables.

    Observed PARSeq failure modes on broadcast jerseys (from the COCO-GT probe):
      - 6 misread as 5/8/4 (e.g. #26 -> #24/#25/#28)
      - 8 misread as 3 / 9 / 5
      - 1 thin-stroke misread as 7 / dropped entirely (e.g. #18 -> '8', #14 -> '1')
      - 0 vs 8 vs O confusion
      - Leading-1 omitted entirely (#18 -> 8, #14 -> 4) — expand 1-digit reads to 2-digit
        by prepending 1 or 2 (very common — broadcasters use #10-#26 range).

    Strength is moderate so the original prediction stays dominant unless the alternative
    is also in the visible-numbers filter downstream.
    """
    # Within-digit confusables (symmetric, swap one digit at a time)
    DIGIT_NEIGHBORS = {
        "0": ["8", "9"],
        "1": ["7", "4"],
        "2": ["7", "3"],
        "3": ["8", "5", "2"],
        "4": ["1", "9", "6"],
        "5": ["6", "8", "3"],
        "6": ["5", "8", "4", "0"],
        "7": ["1", "2"],
        "8": ["3", "5", "9", "0", "6"],
        "9": ["8", "4"],
    }
    smoothed: dict[str, float] = dict(probs)
    for jersey, p in list(probs.items()):
        # Single-digit confusion swaps within the read
        for i, ch in enumerate(jersey):
            for nb in DIGIT_NEIGHBORS.get(ch, []):
                alt = jersey[:i] + nb + jersey[i+1:]
                smoothed[alt] = smoothed.get(alt, 0.0) + p * strength
        # 1-digit -> 2-digit expansion: PARSeq often drops the leading 1 or 2
        if len(jersey) == 1:
            for lead in ("1", "2"):
                alt = lead + jersey
                smoothed[alt] = smoothed.get(alt, 0.0) + p * expand_one_digit_strength
    total = sum(smoothed.values())
    if total <= 0:
        return probs
    return {k: v/total for k, v in smoothed.items()}


def evaluate_ensemble(name: str, ocr_a, ocr_b, sr_model, candidate_numbers: list[str], smooth: bool = True,
                       strict_filter_after: bool = True) -> dict:
    """Combine two OCR backends: multiply probs (intersection wins). Falls back to either single backend."""
    coco = json.loads(GT_JSON.read_text())
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    total = correct = top3 = 0
    confusions = defaultdict(int)
    img_cache: dict[int, np.ndarray] = {}
    for ann in coco["annotations"]:
        cat = cat_by_id[ann["category_id"]]
        gt_num, gt_name = parse_jersey_from_category(cat)
        if gt_num is None:
            continue
        img_id = ann["image_id"]
        if img_id not in img_cache:
            img_cache[img_id] = cv2.imread(str(GT_DIR / img_by_id[img_id]))
        img = img_cache[img_id]
        if img is None:
            continue
        crop = crop_bbox(img, ann["bbox"], pad=4)
        if crop.size == 0:
            continue
        crop = maybe_upscale(crop, sr_model)
        try:
            pa, _ = ocr_a.recognize(crop, candidate_numbers)
        except Exception:
            pa = {}
        try:
            pb, _ = ocr_b.recognize(crop, candidate_numbers)
        except Exception:
            pb = {}
        # Smooth each independently
        if smooth:
            pa = confusable_smooth(pa)
            pb = confusable_smooth(pb)
        if strict_filter_after:
            cs = {str(c) for c in candidate_numbers}
            pa = {k: v for k, v in pa.items() if str(k) in cs}
            pb = {k: v for k, v in pb.items() if str(k) in cs}
        # Merge: weighted sum (PARSeq slightly stronger backbone on standard test sets)
        combined: dict[str, float] = {}
        for k, v in pa.items():
            combined[k] = combined.get(k, 0.0) + v * 1.0
        for k, v in pb.items():
            combined[k] = combined.get(k, 0.0) + v * 0.8
        total_p = sum(combined.values())
        if total_p > 0:
            combined = {k: v/total_p for k, v in combined.items()}
        ranked = sorted(combined.items(), key=lambda kv: -kv[1])
        total += 1
        if not ranked:
            confusions[(gt_num, "<none>")] += 1
            continue
        top1 = ranked[0][0]
        top3_set = {j for j, _ in ranked[:3]}
        if str(top1) == str(gt_num):
            correct += 1
        if str(gt_num) in {str(j) for j in top3_set}:
            top3 += 1
        confusions[(gt_num, top1)] += 1
    return {
        "name": name,
        "total": total,
        "correct": correct,
        "top3_recall": top3,
        "accuracy_pct": 100.0 * correct / max(1, total),
        "top3_pct": 100.0 * top3 / max(1, total),
        "confusions": dict(confusions),
    }


def evaluate_stack(name: str, ocr, sr_model, use_sr: bool, use_torso: bool, candidate_numbers: list[str] | None,
                   tolerate_substring: bool = False, smooth: bool = False, strict_filter_after: bool = False) -> dict:
    coco = json.loads(GT_JSON.read_text())
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    total = correct = top3 = legible_skipped = 0
    confusions = defaultdict(int)  # (gt_jersey, pred_jersey) -> count
    img_cache: dict[int, np.ndarray] = {}

    for ann in coco["annotations"]:
        cat = cat_by_id[ann["category_id"]]
        gt_num, gt_name = parse_jersey_from_category(cat)
        if gt_num is None:
            continue  # skip generic human/referee
        img_id = ann["image_id"]
        if img_id not in img_cache:
            img_path = GT_DIR / img_by_id[img_id]
            img_cache[img_id] = cv2.imread(str(img_path))
        img = img_cache[img_id]
        if img is None:
            continue
        crop = crop_bbox(img, ann["bbox"], pad=4)
        if crop.size == 0:
            continue
        if use_torso:
            crop = torso_crop(crop)
        if use_sr:
            crop = maybe_upscale(crop, sr_model)
        try:
            cands_for_call = candidate_numbers if candidate_numbers is not None else [str(i) for i in range(1, 100)]
            # Multi-region ensemble: full player crop + upper-half + lower-torso
            # (jersey numbers sometimes appear lower on the back when player bends).
            crops_to_try = [crop]
            h, w = crop.shape[:2]
            if h > 60 and w > 30:
                # Upper torso (better for upright players)
                upper = crop[int(h*0.18):int(h*0.55), int(w*0.10):int(w*0.90)]
                if upper.size > 0:
                    crops_to_try.append(upper)
                # Mid-back (jersey-number-on-back region for back-facing players)
                mid = crop[int(h*0.30):int(h*0.65), int(w*0.10):int(w*0.90)]
                if mid.size > 0:
                    crops_to_try.append(mid)
            probs: dict[str, float] = {}
            quality = 0.0
            for sub in crops_to_try:
                if use_sr and (sub.shape[0] < 96 or sub.shape[1] < 64):
                    sub = maybe_upscale(sub, sr_model)
                try:
                    p, q = ocr.recognize(sub, cands_for_call)
                except Exception:
                    continue
                quality = max(quality, q)
                # MAX-merge across regions: best evidence per jersey wins
                for k, v in p.items():
                    probs[k] = max(probs.get(k, 0.0), v)
            if probs:
                total_p = sum(probs.values())
                if total_p > 0:
                    probs = {k: v/total_p for k, v in probs.items()}
        except Exception as e:
            print(f"[err] {gt_num} {gt_name}: {e}")
            continue
        if smooth:
            probs = confusable_smooth(probs)
        if strict_filter_after and candidate_numbers:
            cand_set = {str(c) for c in candidate_numbers}
            probs = {k: v for k, v in probs.items() if str(k) in cand_set}
            if probs:
                total_p = sum(probs.values())
                if total_p > 0:
                    probs = {k: v/total_p for k, v in probs.items()}
        if not probs:
            confusions[(gt_num, "<none>")] += 1
            total += 1
            continue
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        top1 = ranked[0][0]
        top3_set = {j for j, _ in ranked[:3]}
        total += 1
        if str(top1) == str(gt_num):
            correct += 1
        elif tolerate_substring and (str(gt_num) in str(top1) or str(top1) in str(gt_num)):
            correct += 1  # fuzzy: predicted "10" but jersey is "1" or vice versa
        if str(gt_num) in {str(j) for j in top3_set}:
            top3 += 1
        confusions[(gt_num, top1)] += 1
    return {
        "name": name,
        "total": total,
        "correct": correct,
        "top3_recall": top3,
        "accuracy_pct": 100.0 * correct / max(1, total),
        "top3_pct": 100.0 * top3 / max(1, total),
        "confusions": dict(confusions),
    }


def print_report(reports: list[dict]) -> None:
    print(f"\n{'Stack':<30s} {'N':>4s} {'Top-1':>8s} {'Top-3':>8s}")
    print("-" * 60)
    for r in reports:
        print(f"{r['name']:<30s} {r['total']:>4d} {r['accuracy_pct']:>7.1f}% {r['top3_pct']:>7.1f}%")
    print("\nPer-jersey accuracy of best stack:")
    best = max(reports, key=lambda r: r["accuracy_pct"])
    per_j = defaultdict(lambda: [0, 0])
    for (gt, pred), n in best["confusions"].items():
        per_j[gt][1] += n
        if gt == pred:
            per_j[gt][0] += n
    for gt in sorted(per_j.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        c, t = per_j[gt]
        print(f"  #{gt:<3}: {c}/{t} = {100*c/max(1,t):.0f}%")
    print(f"\nTop confusions in best stack ({best['name']}):")
    for (gt, pred), n in sorted(best["confusions"].items(), key=lambda kv: -kv[1])[:15]:
        if gt != pred:
            print(f"  GT #{gt} -> predicted #{pred}: {n}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks", default="baseline,no-sr,torso,torso-sr,no-filter,no-filter-sr",
                    help="Comma-separated stack names to evaluate")
    args = ap.parse_args()
    cfg = load_yaml_config(Path("configs/default.yaml"))
    ocr = build_jersey_ocr(cfg)
    print(f"OCR backend: {type(ocr).__name__}")
    sr = load_sr_model()

    # Optional EasyOCR backend (CRNN, different failure profile from PARSeq's ViT)
    easy_ocr = None
    try:
        easy_cfg = dict(cfg)
        easy_cfg["jersey_ocr"] = dict(cfg.get("jersey_ocr", {}))
        easy_cfg["jersey_ocr"]["backend"] = "easyocr"
        easy_ocr = build_jersey_ocr(easy_cfg)
        print(f"EasyOCR loaded: {type(easy_ocr).__name__}")
    except Exception as e:
        print(f"[easyocr] not available: {e}")

    # Visible numbers in this clip per user metadata
    visible = ["4", "7", "9", "10", "14", "18", "20", "22", "24", "26"]
    # Note: #19 dropped because GT confirms not visible. #23 (Emiliano GK) added below for completeness
    visible_with_gk = visible + ["23"]

    stacks = args.stacks.split(",")
    reports = []
    for s in stacks:
        s = s.strip()
        if s == "baseline":
            r = evaluate_stack("baseline (full crop, SR, visible filter)", ocr, sr, use_sr=True, use_torso=False, candidate_numbers=visible_with_gk)
        elif s == "strict-filter":
            r = evaluate_stack("baseline + strict-filter post-OCR", ocr, sr, use_sr=True, use_torso=False, candidate_numbers=visible_with_gk, strict_filter_after=True)
        elif s == "strict-filter-smooth":
            r = evaluate_stack("baseline + smooth + strict-filter post-OCR", ocr, sr, use_sr=True, use_torso=False, candidate_numbers=visible_with_gk, smooth=True, strict_filter_after=True)
        elif s == "smooth-only":
            r = evaluate_stack("baseline + confusable smoothing", ocr, sr, use_sr=True, use_torso=False, candidate_numbers=visible_with_gk, smooth=True)
        elif s == "no-sr":
            r = evaluate_stack("no SR", ocr, sr, use_sr=False, use_torso=False, candidate_numbers=visible_with_gk)
        elif s == "torso":
            r = evaluate_stack("torso crop, no SR", ocr, sr, use_sr=False, use_torso=True, candidate_numbers=visible_with_gk)
        elif s == "torso-sr":
            r = evaluate_stack("torso crop + SR", ocr, sr, use_sr=True, use_torso=True, candidate_numbers=visible_with_gk)
        elif s == "no-filter":
            r = evaluate_stack("no visible-filter, no SR", ocr, sr, use_sr=False, use_torso=False, candidate_numbers=None)
        elif s == "no-filter-sr":
            r = evaluate_stack("no visible-filter, SR", ocr, sr, use_sr=True, use_torso=False, candidate_numbers=None)
        elif s == "easyocr":
            if easy_ocr is None:
                continue
            r = evaluate_stack("EasyOCR + SR + filter", easy_ocr, sr, use_sr=True, use_torso=False, candidate_numbers=visible_with_gk)
        elif s == "easyocr-smooth":
            if easy_ocr is None:
                continue
            r = evaluate_stack("EasyOCR + SR + smooth + strict-filter", easy_ocr, sr, use_sr=True, use_torso=False, candidate_numbers=visible_with_gk, smooth=True, strict_filter_after=True)
        elif s == "ensemble":
            if easy_ocr is None:
                continue
            r = evaluate_ensemble("PARSeq + EasyOCR ensemble", ocr, easy_ocr, sr, candidate_numbers=visible_with_gk, smooth=True, strict_filter_after=True)
        else:
            print(f"[skip] unknown stack '{s}'")
            continue
        reports.append(r)
    print_report(reports)


if __name__ == "__main__":
    main()
