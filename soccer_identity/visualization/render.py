from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from soccer_identity.utils.schemas import RosterPlayer, Tracklet


def _ascii_normalize(text: str) -> str:
    """OpenCV's Hershey fonts don't support combined Unicode diacritics — "Mbappé" renders
    as "Mbapp??". Strip accents for display only (Mbappé -> Mbappe, Álvarez -> Alvarez).
    """
    if not text:
        return text
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
from soccer_identity.utils.video_io import create_video_writer, get_video_info, iter_video_frames, transcode_to_browser_mp4


@dataclass
class VisualizationRenderer:
    players: list[RosterPlayer]
    confidence_threshold: float = 0.4
    likely_threshold: float = 0.25
    max_seconds: float | None = None
    browser_compatible_mp4: bool = True
    h264_crf: int = 22
    min_player_likelihood: float = 0.25
    min_track_duration_sec: float = 0.25
    min_jersey_display_confidence: float = 0.30
    peak_lock_threshold: float = 0.80
    peak_lock_min_count: int = 2

    def render(
        self,
        video_path: str | Path,
        output_path: str | Path,
        tracklets: list[Tracklet],
        jersey_by_track_id: dict[str, str | None] | None = None,
    ) -> None:
        info = get_video_info(video_path)
        final_output_path = Path(output_path)
        raw_output_path = (
            final_output_path.with_name(f"{final_output_path.stem}.opencv_tmp{final_output_path.suffix}")
            if self.browser_compatible_mp4 and final_output_path.suffix.lower() == ".mp4"
            else final_output_path
        )
        writer = create_video_writer(raw_output_path, info.fps, (info.width, info.height))
        obs_by_frame: dict[int, list[tuple[Tracklet, Any]]] = {}
        for tracklet in tracklets:
            role = tracklet.evidence.get("role") if isinstance(tracklet.evidence, dict) else None
            # Always render GK tracks (they're legit identities); skip referees here only if their
            # duration is too short, but keep them otherwise so we can show "Referee" on screen.
            if role not in {"goalkeeper", "referee"}:
                if not tracklet.is_player or tracklet.player_likelihood < self.min_player_likelihood:
                    continue
            if tracklet.duration < self.min_track_duration_sec:
                continue
            for obs in tracklet.observations:
                obs_by_frame.setdefault(obs.frame_index, []).append((tracklet, obs))
        player_by_id = {player.player_id: player for player in self.players}

        # Build the set of names available via valid (team, jersey) combinations so we
        # can reject snapshot fallbacks where the resolver's earlier guess names a
        # player whose jersey isn't in the team's visible roster (e.g. Guido #18 ARG).
        valid_player_names: set[str] = {_ascii_normalize(p.player_name) for p in self.players}

        # Build a (team, jersey) -> player_name index so the jersey-fallback identity
        # can be derived for tracks that lost resolved_player_id but will still render
        # via the team+jersey lookup.
        player_by_team_jersey: dict[tuple[str, str], str] = {}
        for p in self.players:
            if p.team_name and p.jersey_number is not None:
                player_by_team_jersey[(p.team_name, str(p.jersey_number))] = p.player_name

        def _identity_key(tracklet: Tracklet, jersey_for_track: str | None) -> str | None:
            """Return a (player_name, jersey) identity key normalized so the per-frame
            dedup catches duplicates regardless of whether the label came from
            resolved_player_id, snapshot, or jersey+team fallback. Role-tagged tracks
            (referee/goalkeeper) render as their role and do NOT claim an outfield
            identity — exclude them so they can't suppress real player labels."""
            ev = tracklet.evidence if isinstance(tracklet.evidence, dict) else {}
            if ev.get("role") in {"referee", "goalkeeper"}:
                return None
            if tracklet.resolved_player_id:
                p = player_by_id.get(tracklet.resolved_player_id)
                if p is not None:
                    return f"{_ascii_normalize(p.player_name)}|{p.jersey_number}"
            n = ev.get("resolved_player_name_snapshot")
            j = ev.get("resolved_player_jersey_snapshot")
            if n and j:
                return f"{_ascii_normalize(n)}|{j}"
            # Jersey-fallback path: track has a jersey but no resolved player. The
            # renderer will look up (team_argmax, jersey) -> player_name. Mirror that
            # here so two tracks rendering as the same Messi/Mbappé via fallback are
            # deduplicated.
            if jersey_for_track:
                team_argmax = ev.get("team_argmax")
                if team_argmax:
                    name = player_by_team_jersey.get((team_argmax, str(jersey_for_track)))
                    if name:
                        return f"{_ascii_normalize(name)}|{jersey_for_track}"
            return None

        def _identity_strength(tracklet: Tracklet) -> float:
            """Active resolver wins over snapshots; ties broken by resolved_confidence and duration."""
            base = 1.0 if tracklet.resolved_player_id else 0.0
            return base * 100.0 + float(tracklet.resolved_confidence) * 10.0 + min(10.0, tracklet.duration)

        for frame_index, _timestamp, frame in iter_video_frames(video_path, max_seconds=self.max_seconds):
            overlay = frame.copy()
            entries = obs_by_frame.get(frame_index, [])
            # Per-frame dedup: when multiple tracks claim the same identity (active or
            # snapshot), keep only the strongest. Prevents the two-Mbappé-labels problem
            # introduced when the conflict resolver was relaxed to keep partial-overlap
            # identities alive.
            strength_per_id: dict[str, float] = {}
            for tr, _obs in entries:
                j = jersey_by_track_id.get(tr.track_id) if jersey_by_track_id is not None else None
                k = _identity_key(tr, j)
                if k:
                    strength_per_id[k] = max(strength_per_id.get(k, -1.0), _identity_strength(tr))
            for tracklet, obs in entries:
                j = jersey_by_track_id.get(tracklet.track_id) if jersey_by_track_id is not None else None
                k = _identity_key(tracklet, j)
                if k and _identity_strength(tracklet) < strength_per_id.get(k, 0.0) - 1e-6:
                    cv2.rectangle(overlay, (int(obs.bbox.x1), int(obs.bbox.y1)), (int(obs.bbox.x2), int(obs.bbox.y2)), (96, 96, 96), 1)
                    continue
                self._draw_track(overlay, tracklet, obs, player_by_id, jersey_by_track_id)
            writer.write(overlay)
        writer.release()
        if raw_output_path != final_output_path:
            if transcode_to_browser_mp4(raw_output_path, final_output_path, crf=self.h264_crf):
                raw_output_path.unlink(missing_ok=True)
            else:
                raw_output_path.replace(final_output_path)

    def _draw_track(self, frame: np.ndarray, tracklet: Tracklet, obs: Any, player_by_id: dict[str, RosterPlayer], jersey_by_track_id: dict[str, str | None] | None = None) -> None:
        color = self._track_color(tracklet)
        x1, y1, x2, y2 = map(int, obs.bbox.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = self._label(tracklet, player_by_id, jersey_by_track_id)
        cv2.putText(
            frame,
            label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"ID {tracklet.track_id}",
            (x1, min(frame.shape[0] - 6, y2 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    def _label(self, tracklet: Tracklet, player_by_id: dict[str, RosterPlayer], jersey_by_track_id: dict[str, str | None] | None = None) -> str:
        role = tracklet.evidence.get("role") if isinstance(tracklet.evidence, dict) else None
        if role == "referee":
            return "Referee"
        if role == "goalkeeper":
            return "Goalkeeper"
        if jersey_by_track_id is not None:
            jersey = jersey_by_track_id.get(tracklet.track_id)
        else:
            jersey = self._best_jersey(
                tracklet,
                self.min_jersey_display_confidence,
                self.peak_lock_threshold,
                self.peak_lock_min_count,
            )
        player = player_by_id.get(tracklet.resolved_player_id or "")
        # Resolver gets to display a player name ONLY when OCR confirms the jersey number,
        # OR when its own confidence is very high (e.g. headshot match). Otherwise the
        # resolver tends to attribute "any Argentina-classified track without a number" to
        # a single arbitrary squad player (often Palacios) because team+body alone is too weak.
        ocr_confirms_player = (
            player is not None
            and jersey is not None
            and str(player.jersey_number) == str(jersey)
        )
        if player is not None and tracklet.resolved_confidence >= self.confidence_threshold and ocr_confirms_player:
            return f"{_ascii_normalize(player.player_name)} #{player.jersey_number or '?'} {tracklet.resolved_confidence:.2f}"
        if player is not None and tracklet.resolved_confidence >= self.likely_threshold and ocr_confirms_player:
            return f"Likely {_ascii_normalize(player.player_name)} #{player.jersey_number or '?'} {tracklet.resolved_confidence:.2f}"
        if jersey:
            player_via_jersey = self._player_from_jersey(tracklet, jersey, player_by_id)
            if player_via_jersey is not None:
                return f"{_ascii_normalize(player_via_jersey.player_name)} #{jersey}"
            return f"Low conf ID #{jersey}"
        # v20a: track lost dedup but resolver pinned a player at high confidence.
        if player is not None and tracklet.resolved_confidence >= max(0.55, self.confidence_threshold):
            return f"{_ascii_normalize(player.player_name)} #{player.jersey_number or '?'} {tracklet.resolved_confidence:.2f}"
        # v20b: track lost dedup AND was demoted by conflict resolution. The resolver
        # had pinned a player initially; if the track's top OCR candidate still agrees
        # on that player's jersey, trust the snapshot. Recovers fragments like
        # Mbappé/Upamecano mid-clip where dedup+conflict-resolver nulled resolved_player_id.
        ev = tracklet.evidence if isinstance(tracklet.evidence, dict) else {}
        snapshot_name = ev.get("resolved_player_name_snapshot")
        snapshot_jersey = ev.get("resolved_player_jersey_snapshot")
        if snapshot_name and snapshot_jersey:
            cands = ev.get("jersey_candidates") if isinstance(ev.get("jersey_candidates"), list) else None
            if cands:
                try:
                    top_j, top_s = cands[0]
                    if str(top_j) == str(snapshot_jersey) and float(top_s) >= 5.0:
                        return f"Likely {_ascii_normalize(snapshot_name)} #{snapshot_jersey}"
                except Exception:
                    pass
        return "Low conf ID"

    def _player_from_jersey(
        self,
        tracklet: Tracklet,
        jersey: str,
        player_by_id: dict[str, RosterPlayer],
    ) -> RosterPlayer | None:
        matches = [p for p in player_by_id.values() if str(p.jersey_number) == str(jersey)]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        # Multiple teams share this jersey -> use the tracklet's team argmax (stamped by
        # _stamp_team_argmax) to pick the right one (e.g. #10 -> Messi vs Mbappé).
        team_argmax = tracklet.evidence.get("team_argmax") if isinstance(tracklet.evidence, dict) else None
        if team_argmax:
            for p in matches:
                if p.team_name == team_argmax:
                    return p
        return matches[0]

    @staticmethod
    def _best_jersey(
        tracklet: Tracklet,
        min_confidence: float = 0.40,
        peak_threshold: float = 0.85,
        peak_min_count: int = 4,
        longer_lock_min_count: int = 3,
    ) -> str | None:
        votes: dict[str, float] = {}
        peak_counts: dict[str, int] = {}
        for obs in tracklet.observations:
            for jersey, prob in obs.jersey_probs.items():
                contribution = prob * max(0.01, obs.jersey_quality)
                votes[jersey] = votes.get(jersey, 0.0) + contribution
                if contribution >= peak_threshold:
                    peak_counts[jersey] = peak_counts.get(jersey, 0) + 1
        if not votes:
            return None
        total = sum(votes.values())
        if total <= 0:
            return None
        weighted_best = max(votes, key=votes.get)
        weighted_share = votes[weighted_best] / total
        # If a longer peak-locked candidate contains the weighted winner, prefer the longer one.
        longer = [j for j, c in peak_counts.items() if c >= longer_lock_min_count and len(j) > len(weighted_best) and weighted_best in j]
        if longer:
            return max(longer, key=lambda j: (peak_counts[j], votes[j]))
        if peak_counts.get(weighted_best, 0) >= peak_min_count:
            return weighted_best
        if weighted_share >= min_confidence:
            return weighted_best
        return None

    @staticmethod
    def _track_color(tracklet: Tracklet) -> tuple[int, int, int]:
        seed = int(tracklet.track_id) if tracklet.track_id.isdigit() else abs(hash(tracklet.track_id))
        colors = [
            (58, 180, 255),
            (72, 214, 111),
            (255, 170, 68),
            (220, 96, 255),
            (92, 234, 225),
            (255, 96, 112),
        ]
        return colors[seed % len(colors)]



def build_renderer(config: dict[str, Any], players: list[RosterPlayer]) -> VisualizationRenderer:
    viz_config = config.get("visualization", {})
    assignment_config = config.get("assignment", {})
    return VisualizationRenderer(
        players=players,
        confidence_threshold=float(assignment_config.get("identity_confidence_threshold", 0.4)),
        likely_threshold=float(viz_config.get("likely_threshold", 0.25)),
        max_seconds=viz_config.get("max_seconds"),
        browser_compatible_mp4=bool(viz_config.get("browser_compatible_mp4", True)),
        h264_crf=int(viz_config.get("h264_crf", 22)),
        min_player_likelihood=float(viz_config.get("min_player_likelihood", 0.25)),
        min_track_duration_sec=float(viz_config.get("min_track_duration_sec", 0.25)),
        min_jersey_display_confidence=float(viz_config.get("min_jersey_display_confidence", 0.30)),
        peak_lock_threshold=float(viz_config.get("peak_lock_threshold", 0.80)),
        peak_lock_min_count=int(viz_config.get("peak_lock_min_count", 2)),
    )
