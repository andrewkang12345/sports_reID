"""Evaluate a run's output against the saved ARG-FRA ground truth.

Usage:
    python3 eval_against_gt.py outputs/clip_ARG_FRA_183303_v13 [v14 v15 ...]

Reports per-version metrics:
- Recall: how many GT players have any track displaying their name
- Precision: of displayed labels, how many are GT players (no off-roster false positives)
- Coverage seconds per GT player
- False positives (players labeled but not in GT)
- Track fragmentation per player (lower is better — less label teleport)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_gt(gt_path: str = "outputs/ground_truth_arg_fra.json") -> dict[str, set[str]]:
    """Return {team_name -> set of player_names} expected to be visible."""
    data = json.loads(Path(gt_path).read_text())
    out: dict[str, set[str]] = {}
    for team, info in data["expected_visible"].items():
        if isinstance(info, dict) and "players" in info:
            out[team] = {p["name"] for p in info["players"]}
    out["Referees"] = data["expected_visible"].get("Referees", 0)
    out["Goalkeepers"] = data["expected_visible"].get("Goalkeepers", 0)
    return out


def render_label(t: dict[str, Any], player_by_id: dict[str, dict]) -> str | None:
    role = t.get("role")
    if role == "referee":
        return "Referee"
    if role == "goalkeeper":
        return "Goalkeeper"
    j = t.get("best_jersey_guess")
    rp = t.get("resolved_player")
    conf = t.get("resolved_confidence", 0)
    ocr_confirms = rp and j and str(rp["jersey_number"]) == str(j)
    if rp and conf >= 0.4 and ocr_confirms:
        return f"{rp['player_name']} #{rp['jersey_number']}"
    if rp and conf >= 0.25 and ocr_confirms:
        return f"Likely {rp['player_name']} #{rp['jersey_number']}"
    if j:
        # An appearance-memory override may have rewritten team_argmax to the cluster's
        # canonical team — respect that over the noisy raw team_probs.
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
                    return f"{p['player_name']} #{j}"
        if len(cands) == 1:
            return f"{cands[0]['player_name']} #{j}"
    return None


def evaluate(run_dir: str, gt: dict) -> dict:
    rdir = Path(run_dir)
    res = json.loads((rdir / "result.json").read_text())
    debug = json.loads((rdir / "debug_tracks.json").read_text())
    meta = json.loads((rdir / "metadata.json").read_text())
    player_by_id: dict[str, dict] = {}
    for team, roster in meta["rosters"].items():
        for p in roster:
            pid = f"{team}|{p['jersey_number']}|{p['player_name']}"
            player_by_id[pid] = p
    debug_by_tid = {tr["track_id"]: tr for tr in debug}

    # Match the renderer's min_track_duration_sec filter (0.25s by default). Tracks shorter
    # than this never appear on screen, so they shouldn't count as Extra in the eval either.
    min_render_duration = 0.25
    by_label: dict[str, set[int]] = defaultdict(set)
    label_tracks: dict[str, list[str]] = defaultdict(list)
    for t in res["tracks"]:
        duration = float(t.get("duration", 0.0))
        if duration < min_render_duration:
            continue
        label = render_label(t, player_by_id)
        if not label or label.startswith("Low conf"):
            continue
        name = label.split(" #")[0].replace("Likely ", "")
        label_tracks[name].append(t["track_id"])
        tr = debug_by_tid.get(t["track_id"])
        if tr:
            for s in range(int(tr["start_time"]), int(tr["end_time"]) + 1):
                by_label[name].add(s)

    # Expected GT set (player names + role tags)
    expected = set()
    for team, names in gt.items():
        if isinstance(names, set):
            expected |= names
    expected_referees = gt.get("Referees", 0)
    expected_goalkeepers = gt.get("Goalkeepers", 0)

    displayed = set(by_label.keys()) - {"Referee", "Goalkeeper"}
    hit = displayed & expected
    missed = expected - displayed
    extra = displayed - expected

    referee_seen = "Referee" in by_label
    goalkeeper_seen = "Goalkeeper" in by_label

    # Fragmentation total (lower = less teleport)
    total_fragments = sum(len(tracks) for name, tracks in label_tracks.items() if name in expected)

    return {
        "run": str(rdir),
        "n_expected": len(expected),
        "n_displayed": len(displayed),
        "n_hit": len(hit),
        "n_missed": len(missed),
        "n_extra": len(extra),
        "recall": len(hit) / max(1, len(expected)),
        "precision": len(hit) / max(1, len(displayed)) if displayed else 0,
        "hit": sorted(hit),
        "missed": sorted(missed),
        "extra": sorted(extra),
        "coverage_sec": {n: len(by_label.get(n, set())) for n in expected},
        "fragments": {n: len(label_tracks.get(n, [])) for n in expected},
        "total_fragments_gt_players": total_fragments,
        "referee_seen": referee_seen,
        "expected_referees": expected_referees,
        "goalkeeper_seen": goalkeeper_seen,
        "expected_goalkeepers": expected_goalkeepers,
    }


def print_report(report: dict) -> None:
    r = report
    print(f"\n=== {r['run']} ===")
    print(f"  Recall: {r['n_hit']}/{r['n_expected']} ({r['recall']:.1%}); Precision: {r['n_hit']}/{r['n_displayed']} ({r['precision']:.1%})")
    print(f"  Hit ({len(r['hit'])}):     {', '.join(r['hit'])}")
    print(f"  Missed ({len(r['missed'])}):  {', '.join(r['missed']) or 'none'}")
    print(f"  Extra ({len(r['extra'])}):   {', '.join(r['extra']) or 'none'}")
    print(f"  Referee: {'YES' if r['referee_seen'] else 'NO'} (expected: {r['expected_referees']}+)")
    print(f"  Goalkeeper: {'YES' if r['goalkeeper_seen'] else 'NO'} (expected: {r['expected_goalkeepers']}+)")
    print(f"  Total fragments across GT players: {r['total_fragments_gt_players']} (lower = less teleport)")
    print(f"  Coverage per GT player (sec):")
    for n, s in sorted(r["coverage_sec"].items(), key=lambda kv: -kv[1]):
        frags = r["fragments"].get(n, 0)
        print(f"    {n:<24s} {s:>3}s  (fragments: {frags})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--gt", default="outputs/ground_truth_arg_fra.json")
    args = parser.parse_args()
    gt = load_gt(args.gt)
    for d in args.dirs:
        try:
            r = evaluate(d, gt)
            print_report(r)
        except FileNotFoundError as e:
            print(f"\n=== {d} ===\n  ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
