"""Cross-track appearance memory built from high-confidence OCR moments.

For each tracklet, find the K frames where PARSeq read a jersey number with the highest
combined confidence (top probability × jersey_quality). Snapshot the ReID embedding at
those frames as "anchors", tagged with the candidate (team, jersey, OCR score). Then
cluster anchors across tracks by cosine distance to produce a small set of putative
player identities; the cluster's canonical jersey is the highest-scoring anchor's vote.

This is an *override* step that runs AFTER the existing dedup. The intent: catch cases
like Mbappé starting labeled #10, then mid-clip getting re-labeled #4 because PARSeq
misread a single frame, or a fallen player being labeled "#24" because a few low-quality
frames spelled out "2" or "4". The anchor (= the high-confidence moment) wins.

Editable: a track's identity flips if a *more confident* anchor on its cluster mate
appears later. The cluster's canonical jersey is recomputed by max-conf each merge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from soccer_identity.utils.schemas import Tracklet


@dataclass
class Anchor:
    """One high-confidence OCR moment on one track."""
    track_id: str
    frame_index: int
    embedding: np.ndarray  # (D,), L2-normalized
    jersey: str
    team: str | None
    score: float  # top_prob * jersey_quality at this frame


@dataclass
class IdentityCluster:
    cluster_id: int
    anchors: list[Anchor] = field(default_factory=list)
    track_ids: set[str] = field(default_factory=set)
    embedding: np.ndarray | None = None  # weighted centroid

    @property
    def best_anchor(self) -> Anchor | None:
        if not self.anchors:
            return None
        return max(self.anchors, key=lambda a: a.score)


def _per_track_anchors(
    tracklet: Tracklet,
    team_of_track: str | None,
    score_threshold: float,
    k_per_jersey: int,
) -> list[Anchor]:
    """Find this track's top-K OCR frames per distinct jersey hypothesis.

    A track might have peaks on multiple jerseys (PARSeq's "10" bias means most tracks
    have a "10" peak). Keep anchors for ALL jersey hypotheses scoring above threshold —
    the cluster step will resolve which one wins via cross-track agreement.
    """
    if not tracklet.observations:
        return []
    appearance_field = "appearance_embedding"
    # Score every (jersey, obs) pair
    per_jersey: dict[str, list[tuple[float, Any]]] = {}
    for obs in tracklet.observations:
        emb = getattr(obs, appearance_field, None)
        if emb is None:
            continue
        for j, p in (obs.jersey_probs or {}).items():
            score = float(p) * float(obs.jersey_quality or 0.0)
            if score < score_threshold:
                continue
            per_jersey.setdefault(j, []).append((score, obs))
    anchors: list[Anchor] = []
    for jersey, scored in per_jersey.items():
        scored.sort(reverse=True, key=lambda x: x[0])
        for score, obs in scored[:k_per_jersey]:
            emb_list = getattr(obs, appearance_field, None)
            if emb_list is None:
                continue
            emb = np.asarray(emb_list, dtype=np.float32)
            n = float(np.linalg.norm(emb))
            if n < 1e-8:
                continue
            emb = emb / n
            anchors.append(
                Anchor(
                    track_id=tracklet.track_id,
                    frame_index=int(obs.frame_index),
                    embedding=emb,
                    jersey=jersey,
                    team=team_of_track,
                    score=score,
                )
            )
    return anchors


def cluster_anchors(
    anchors: list[Anchor],
    cosine_threshold: float = 0.80,
    track_active_frames: dict[str, set[int]] | None = None,
) -> list[IdentityCluster]:
    """Greedy single-link clustering by cosine similarity.

    Anchors are processed in descending score order: the highest-score anchors define
    cluster seeds, and subsequent anchors merge into the nearest existing cluster if
    similarity exceeds threshold AND the cluster's dominant jersey is compatible AND
    the candidate track does not temporally overlap any other track already in the
    cluster (a single player cannot be in two places at once).
    """
    sorted_anchors = sorted(anchors, key=lambda a: -a.score)
    clusters: list[IdentityCluster] = []
    active = track_active_frames or {}
    for a in sorted_anchors:
        best_idx = -1
        best_sim = -1.0
        for idx, c in enumerate(clusters):
            if c.embedding is None:
                continue
            # Reject opposite-team merges
            best = c.best_anchor
            if best is not None and a.team and best.team and a.team != best.team:
                continue
            # Reject merge if candidate track temporally overlaps any other track in this cluster
            cand_frames = active.get(a.track_id, set())
            if cand_frames:
                conflict = False
                for tid in c.track_ids:
                    if tid == a.track_id:
                        continue
                    other_frames = active.get(tid, set())
                    if other_frames and (cand_frames & other_frames):
                        conflict = True
                        break
                if conflict:
                    continue
            sim = float(np.dot(a.embedding, c.embedding))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim >= cosine_threshold:
            c = clusters[best_idx]
            c.anchors.append(a)
            c.track_ids.add(a.track_id)
            embs = np.stack([x.embedding for x in c.anchors], axis=0)
            weights = np.asarray([x.score for x in c.anchors], dtype=np.float32)
            mix = (embs * weights[:, None]).sum(axis=0)
            n = float(np.linalg.norm(mix))
            if n > 1e-8:
                mix /= n
            c.embedding = mix.astype(np.float32)
        else:
            clusters.append(
                IdentityCluster(
                    cluster_id=len(clusters),
                    anchors=[a],
                    track_ids={a.track_id},
                    embedding=a.embedding.copy(),
                )
            )
    return clusters


def assign_canonical_jersey(
    cluster: IdentityCluster,
    team_rosters: dict[str, set[str]] | None = None,
) -> tuple[str | None, str | None, float]:
    """Within a cluster, vote on the jersey using anchor scores. Returns (jersey, team, score).

    The canonical team is taken from the highest-scoring anchor for the winning jersey,
    NOT averaged across anchors — the strongest anchor usually has the cleanest crop and
    thus the most reliable kit-color read.

    If `team_rosters` is provided (visible_numbers per team), the canonical jersey must be
    in the cluster-team's visible roster — otherwise we fall through to the next-best
    jersey in the cluster that IS valid. This rejects e.g. an Argentine cluster forcing
    #18 (PARSeq misread of #19) when Argentina has no #18.
    """
    if not cluster.anchors:
        return None, None, 0.0
    by_jersey: dict[str, float] = {}
    best_anchor_for_jersey: dict[str, Anchor] = {}
    for a in cluster.anchors:
        by_jersey[a.jersey] = by_jersey.get(a.jersey, 0.0) + a.score
        prev = best_anchor_for_jersey.get(a.jersey)
        if prev is None or a.score > prev.score:
            best_anchor_for_jersey[a.jersey] = a
    ranked = sorted(by_jersey.items(), key=lambda kv: -kv[1])
    for jersey, total_score in ranked:
        canonical_team = best_anchor_for_jersey[jersey].team
        if team_rosters and canonical_team:
            roster = team_rosters.get(canonical_team)
            if roster is not None and str(jersey) not in roster:
                continue  # not visible for this team — try next-best jersey
        return jersey, canonical_team, total_score
    return None, None, 0.0


def build_appearance_memory(
    tracklets: list[Tracklet],
    team_of_track: dict[str, str | None],
    excluded_track_ids: set[str] | None = None,
    score_threshold: float = 0.70,
    k_per_jersey: int = 3,
    cosine_threshold: float = 0.80,
) -> tuple[list[IdentityCluster], dict[str, int]]:
    """Cluster all tracks across the clip by appearance.

    Returns (clusters, track_to_cluster_id).
    Tracks in `excluded_track_ids` (e.g. role-tagged) are skipped — they should not
    contribute jersey identities.
    """
    excluded = set(excluded_track_ids or [])
    all_anchors: list[Anchor] = []
    track_active_frames: dict[str, set[int]] = {}
    for tr in tracklets:
        if tr.track_id in excluded:
            continue
        team = team_of_track.get(tr.track_id)
        anchors = _per_track_anchors(tr, team, score_threshold, k_per_jersey)
        all_anchors.extend(anchors)
        track_active_frames[tr.track_id] = {int(obs.frame_index) for obs in tr.observations}
    clusters = cluster_anchors(
        all_anchors, cosine_threshold=cosine_threshold, track_active_frames=track_active_frames
    )
    track_to_cluster: dict[str, int] = {}
    for c in clusters:
        for tid in c.track_ids:
            # If multiple clusters claim the same track, keep the one with highest best-anchor score
            if tid in track_to_cluster:
                existing = clusters[track_to_cluster[tid]]
                ex_best = existing.best_anchor.score if existing.best_anchor else 0.0
                cur_best = c.best_anchor.score if c.best_anchor else 0.0
                if cur_best <= ex_best:
                    continue
            track_to_cluster[tid] = c.cluster_id
    return clusters, track_to_cluster


def override_jerseys_via_memory(
    track_summaries: list[dict],
    tracklets_by_id: dict[str, Tracklet],
    clusters: list[IdentityCluster],
    track_to_cluster: dict[str, int],
    min_override_margin: float = 0.15,
    team_rosters: dict[str, set[str]] | None = None,
) -> int:
    """For each cluster, pick a canonical jersey and overwrite weaker tracks.

    The override fires when:
      - the cluster has at least 2 distinct tracks (so we're propagating evidence
        from a confidently-labeled track to a sibling, not making things up)
      - the cluster's best anchor score exceeds the track's own best score for its
        current jersey by a margin (default 0.15) — protects strong existing labels
      - the track has no resolved player attached yet OR the current jersey differs

    Editable: a later, stronger anchor on the same cluster wins because `best_anchor`
    is recomputed each call from the cluster's anchor list.
    """
    summary_by_id = {t["track_id"]: t for t in track_summaries}
    changed = 0
    for c in clusters:
        if len(c.track_ids) < 2:
            # A single-track cluster cannot meaningfully "override" itself — its label
            # already came from its own OCR. Skip to avoid no-op renames.
            continue
        jersey, team, _ = assign_canonical_jersey(c, team_rosters=team_rosters)
        if not jersey:
            continue
        best = c.best_anchor
        if best is None:
            continue
        anchor_score = best.score
        for tid in c.track_ids:
            summary = summary_by_id.get(tid)
            if summary is None:
                continue
            current = summary.get("best_jersey_guess")
            tr = tracklets_by_id.get(tid)
            own_score = 0.0
            if tr is not None and current:
                for obs in tr.observations:
                    if not obs.jersey_probs:
                        continue
                    p = float(obs.jersey_probs.get(str(current), 0.0))
                    s = p * float(obs.jersey_quality or 0.0)
                    if s > own_score:
                        own_score = s
            # Margin gate: cluster anchor must beat this track's own best by `min_override_margin`.
            # This protects strong existing labels — only the weakest tracks (or those with the
            # wrong jersey) get rewritten.
            if str(current) != str(jersey) and anchor_score > own_score + min_override_margin:
                summary["best_jersey_guess"] = jersey
                summary["jersey_override_source"] = "appearance_memory"
                summary["jersey_override_anchor_score"] = round(float(anchor_score), 3)
                summary["jersey_override_cluster_id"] = int(c.cluster_id)
                # Also override the team_argmax used by the renderer's team+jersey fallback.
                # Without this, a French #18 (Upamecano) track whose color was misread can get
                # rendered as "Guido Rodríguez #18" (the Argentine #18) because team_probs
                # narrowly favored Argentina. The cluster's canonical team is the team of the
                # highest-confidence anchor — that's the trustworthy signal.
                if team:
                    summary["team_probs_override"] = {team: 1.0}
                    summary["team_argmax_override"] = team
                if tr is not None:
                    tr.evidence["best_jersey_guess"] = jersey
                    tr.evidence["jersey_override_source"] = "appearance_memory"
                    tr.evidence["jersey_override_anchor_score"] = round(float(anchor_score), 3)
                    if team:
                        tr.evidence["team_argmax"] = team
                        tr.evidence["team_argmax_override"] = team
                changed += 1
    return changed
