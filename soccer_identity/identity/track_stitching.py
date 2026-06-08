"""Aggressive cross-track identity stitching.

Premise (from real broadcast clips): a player's jersey number is visible for only a
handful of frames within their on-screen duration. To label them across the WHOLE clip
we need to STITCH tracker fragments of the same physical player into a single identity
group, then propagate any number reading from one fragment to the whole group.

Algorithm:
  1. For each track, compute a per-track appearance fingerprint (mean OSNet embedding
     across all observations) and per-track team argmax.
  2. Build an Identity Graph over tracks. Add an edge A--B when:
       - Same team (or one is team-unknown).
       - Temporal disjoint OR brief overlap (<= max_overlap_sec).
       - Spatial trajectory compatible (predicted-end-of-A near start-of-B if disjoint,
         OR center-distance small if overlapping briefly).
       - Appearance cosine >= app_threshold.
       - No CONTRADICTORY OCR (their top jersey candidates are compatible).
  3. Union-find merge connected components -> identity groups.
  4. For each group, pick a canonical identity:
       - Highest resolved_confidence wins.
       - If no resolver, highest jersey_vote_score wins.
       - Apply the canonical identity to ALL tracks in the group.

This propagates jersey/identity from the few number-readable frames across all
fragments of the same physical player.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _avg_embedding(obs_list) -> np.ndarray | None:
    embs = []
    for obs in obs_list:
        e = getattr(obs, "appearance_embedding", None)
        if e is None:
            continue
        embs.append(np.asarray(e, dtype=np.float32))
    if not embs:
        return None
    avg = np.mean(np.stack(embs, axis=0), axis=0)
    n = float(np.linalg.norm(avg))
    return avg / max(1e-8, n)


def _team_of(t: dict) -> str | None:
    tp = t.get("team_probs") or {}
    return max(tp, key=tp.get) if tp else None


def _last_pos(tracklet) -> tuple[float, float] | None:
    if not tracklet.observations:
        return None
    obs = max(tracklet.observations, key=lambda o: o.frame_index)
    x1, y1, x2, y2 = obs.bbox.xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _first_pos(tracklet) -> tuple[float, float] | None:
    if not tracklet.observations:
        return None
    obs = min(tracklet.observations, key=lambda o: o.frame_index)
    x1, y1, x2, y2 = obs.bbox.xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _top_jersey(t: dict) -> str | None:
    cands = t.get("jersey_candidates") or []
    return str(cands[0][0]) if cands else None


class _DSU:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def stitch_tracks(
    tracks_summary: list[dict],
    tracklets: list[Any],
    player_by_id: dict[str, Any],
    metadata: dict | None = None,
    app_threshold: float = 0.65,
    max_gap_sec: float = 3.0,
    max_overlap_sec: float = 1.0,
    max_center_dist: float = 250.0,
) -> dict:
    """Merge same-player track fragments. Mutates tracks_summary + tracklets in place.

    Returns a stats dict.
    """
    by_tid = {tr.track_id: tr for tr in tracklets}
    # Exclude role tracks from stitching; they're not outfield players
    elig: list[dict] = [t for t in tracks_summary if t.get("is_player", True) and t.get("role") not in {"referee", "goalkeeper"}]
    if len(elig) < 2:
        return {"groups": 0, "stitched_pairs": 0, "propagated_tracks": 0}

    # Per-track appearance fingerprints
    fp: dict[str, np.ndarray] = {}
    for t in elig:
        tr = by_tid.get(t["track_id"])
        if tr is None:
            continue
        avg = _avg_embedding(tr.observations)
        if avg is not None:
            fp[t["track_id"]] = avg

    # Per-track visible-roster validity for OCR jerseys (so we don't propagate Guido #18 ARG)
    visible_for_team: dict[str, set[str]] = {}
    if metadata:
        vm = metadata.get("visible_jersey_numbers") or {}
        visible_for_team = {team: {str(n) for n in nums} for team, nums in vm.items()}

    dsu = _DSU([t["track_id"] for t in elig])
    edges = 0
    for i, A in enumerate(elig):
        a_tid = A["track_id"]
        a_tr = by_tid.get(a_tid)
        if a_tr is None or a_tid not in fp:
            continue
        a_team = _team_of(A)
        a_jersey = _top_jersey(A) or A.get("best_jersey_guess")
        a_emb = fp[a_tid]
        a_end_pos = _last_pos(a_tr)
        a_start_pos = _first_pos(a_tr)
        for B in elig[i + 1:]:
            b_tid = B["track_id"]
            if b_tid not in fp:
                continue
            b_tr = by_tid.get(b_tid)
            if b_tr is None:
                continue
            # Team consistency (if both known)
            b_team = _team_of(B)
            if a_team and b_team and a_team != b_team:
                continue
            # Temporal: disjoint OR brief overlap only
            gap_a_to_b = b_tr.start_time - a_tr.end_time
            gap_b_to_a = a_tr.start_time - b_tr.end_time
            overlap = -max(gap_a_to_b, gap_b_to_a)
            if overlap > max_overlap_sec:
                continue
            if gap_a_to_b > max_gap_sec or gap_b_to_a > max_gap_sec:
                continue
            # Spatial: A's last pos should be near B's first pos (or vice versa)
            from soccer_identity.utils.geometry import point_distance
            spatial_ok = True
            if gap_a_to_b > 0 and a_end_pos and _first_pos(b_tr):
                if point_distance(a_end_pos, _first_pos(b_tr)) > max_center_dist:
                    spatial_ok = False
            elif gap_b_to_a > 0 and a_start_pos and _last_pos(b_tr):
                if point_distance(a_start_pos, _last_pos(b_tr)) > max_center_dist:
                    spatial_ok = False
            if not spatial_ok:
                continue
            # OCR contradiction check: if both have a confident top jersey AND they
            # differ, AND both are visible-roster-valid, reject.
            b_jersey = _top_jersey(B) or B.get("best_jersey_guess")
            if a_jersey and b_jersey and a_jersey != b_jersey:
                # Allow if one of them isn't valid for the team (likely a misread)
                team = a_team or b_team
                roster = visible_for_team.get(team) if team else None
                if roster is not None:
                    a_valid = str(a_jersey) in roster
                    b_valid = str(b_jersey) in roster
                    if a_valid and b_valid:
                        continue
                else:
                    continue
            # Appearance similarity
            b_emb = fp[b_tid]
            sim = float(np.dot(a_emb, b_emb))
            if sim < app_threshold:
                continue
            dsu.union(a_tid, b_tid)
            edges += 1

    # Build groups
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in elig:
        groups[dsu.find(t["track_id"])].append(t)

    # For each group with >1 member, propagate the strongest identity
    n_groups_merged = 0
    n_propagated = 0
    for root, members in groups.items():
        if len(members) < 2:
            continue
        # Pick best identity:
        #   Prefer a member with active resolved_player_id and highest confidence
        #   Else prefer a member with confident jersey + team mapping to a known player
        best_resolved_id = None
        best_conf = 0.0
        best_jersey = None
        best_jersey_score = 0.0
        best_team = None
        for t in members:
            rpid = t.get("resolved_player_id")
            conf = float(t.get("resolved_confidence", 0.0) or 0.0)
            if rpid and conf >= best_conf:
                best_resolved_id = rpid
                best_conf = conf
            jscore = float(t.get("jersey_vote_score", 0.0) or 0.0)
            jersey = t.get("best_jersey_guess")
            if jersey and jscore > best_jersey_score:
                best_jersey = jersey
                best_jersey_score = jscore
                best_team = _team_of(t)
        # If we have a resolved id, look up its (team, jersey) for consistency check
        rp_team = rp_jersey = None
        if best_resolved_id:
            rp = player_by_id.get(best_resolved_id)
            if rp is not None:
                rp_team = rp.team_name
                rp_jersey = str(rp.jersey_number) if rp.jersey_number is not None else None
        # Decide canonical group identity
        canonical_pid = best_resolved_id
        canonical_jersey = best_jersey
        canonical_team = best_team or rp_team
        # If only jersey + team known (no resolver), map to player_id
        if canonical_pid is None and canonical_jersey and canonical_team:
            for pid, p in player_by_id.items():
                if p.team_name == canonical_team and str(p.jersey_number) == str(canonical_jersey):
                    canonical_pid = pid
                    break
        if not canonical_pid and not canonical_jersey:
            continue  # no identity to propagate
        # Propagate
        n_groups_merged += 1
        for t in members:
            changed = False
            if canonical_pid and not t.get("resolved_player_id"):
                t["resolved_player_id"] = canonical_pid
                t["resolved_confidence"] = max(float(t.get("resolved_confidence", 0.0) or 0.0), 0.3)
                t["identity_stitched_from"] = root
                if canonical_pid in player_by_id:
                    t["resolved_player"] = player_by_id[canonical_pid].to_dict()
                # Mirror onto live tracklet
                tr = by_tid.get(t["track_id"])
                if tr is not None:
                    tr.resolved_player_id = canonical_pid
                    tr.resolved_confidence = max(tr.resolved_confidence, 0.3)
                changed = True
            if canonical_jersey and not t.get("best_jersey_guess"):
                t["best_jersey_guess"] = canonical_jersey
                t["jersey_stitched_from"] = root
                changed = True
            if changed:
                n_propagated += 1
    # NOTE: previous experiments (v37/v38/v39/v40) tried to propagate the canonical
    # roster jersey across all tracks of a resolved player. This propagated resolver
    # mistakes (Molina mis-resolved as Enzo, etc.) and regressed detection benchmark
    # by ~20pp. Disabled. v32 (stitcher only, no jersey override) remains best.
    n_player_propagated = 0
    return {
        "groups": n_groups_merged,
        "stitched_pairs": edges,
        "propagated_tracks": n_propagated,
        "player_jersey_propagated": n_player_propagated,
        "total_eligible_tracks": len(elig),
        "tracks_with_fp": len(fp),
    }
