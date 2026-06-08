from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from soccer_identity.identity.body_reid import BodyReIDMemory
from soccer_identity.utils.schemas import RosterPlayer, Tracklet

try:  # pragma: no cover - optional dependency branch
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class WindowHungarianAssigner:
    players: list[RosterPlayer]
    body_memory: BodyReIDMemory
    identity_confidence_threshold: float = 0.4
    window_sec: float = 1.0
    step_sec: float = 0.5

    def assign(self, tracklets: list[Tracklet]) -> None:
        player_by_id = {player.player_id: player for player in self.players}
        players_by_team: dict[str, list[RosterPlayer]] = {}
        for player in self.players:
            players_by_team.setdefault(player.team_name, []).append(player)

        for tracklet in tracklets:
            if not tracklet.identity_posterior:
                continue
            best_id = max(tracklet.identity_posterior, key=tracklet.identity_posterior.get)
            tracklet.resolved_player_id = best_id
            tracklet.resolved_confidence = float(tracklet.identity_posterior[best_id])

        if not tracklets:
            return
        start = min(t.start_time for t in tracklets)
        end = max(t.end_time for t in tracklets)
        votes: dict[str, dict[str, float]] = {tracklet.track_id: {} for tracklet in tracklets}
        t = start
        while t <= end + 1e-6:
            active = [tracklet for tracklet in tracklets if tracklet.start_time <= t + self.window_sec and tracklet.end_time >= t]
            self._hungarian_window(active, players_by_team, player_by_id, votes, t, t + self.window_sec)
            self._resolve_conflicts(active, player_by_id)
            t += self.step_sec

        for tracklet in tracklets:
            track_votes = votes.get(tracklet.track_id, {})
            if not track_votes:
                continue
            selected = max(track_votes, key=track_votes.get)
            tracklet.resolved_player_id = selected
            tracklet.resolved_confidence = float(tracklet.identity_posterior.get(selected, 0.0))

        for tracklet in sorted(tracklets, key=lambda item: item.start_time):
            if tracklet.resolved_player_id and tracklet.resolved_confidence >= self.identity_confidence_threshold:
                self.body_memory.update(tracklet, tracklet.resolved_player_id)

    def _hungarian_window(
        self,
        active: list[Tracklet],
        players_by_team: dict[str, list[RosterPlayer]],
        player_by_id: dict[str, RosterPlayer],
        votes: dict[str, dict[str, float]],
        window_start: float,
        window_end: float,
    ) -> None:
        grouped: dict[str, list[Tracklet]] = {}
        for tracklet in active:
            team = self._dominant_team(tracklet, player_by_id)
            if team is None:
                continue
            grouped.setdefault(team, []).append(tracklet)

        for team, team_tracklets in grouped.items():
            candidates = players_by_team.get(team, [])
            if not candidates:
                continue
            cost = np.ones((len(team_tracklets), len(candidates)), dtype=np.float32)
            for row, tracklet in enumerate(team_tracklets):
                for col, player in enumerate(candidates):
                    prob = float(tracklet.identity_posterior.get(player.player_id, 0.0))
                    cost[row, col] = 1.0 - prob
            if linear_sum_assignment is not None:
                rows, cols = linear_sum_assignment(cost)
                assignments = zip(rows.tolist(), cols.tolist())
            else:  # pragma: no cover
                assignments = self._greedy_assign(cost)
            for row, col in assignments:
                tracklet = team_tracklets[row]
                player = candidates[col]
                prob = float(tracklet.identity_posterior.get(player.player_id, 0.0))
                if prob < max(0.12, self.identity_confidence_threshold * 0.35):
                    continue
                overlap = max(0.0, min(tracklet.end_time, window_end) - max(tracklet.start_time, window_start))
                votes.setdefault(tracklet.track_id, {})
                votes[tracklet.track_id][player.player_id] = votes[tracklet.track_id].get(player.player_id, 0.0) + prob * max(overlap, 0.05)

    @staticmethod
    def _greedy_assign(cost: np.ndarray) -> list[tuple[int, int]]:
        flat = sorted((float(cost[row, col]), row, col) for row in range(cost.shape[0]) for col in range(cost.shape[1]))
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        out: list[tuple[int, int]] = []
        for _value, row, col in flat:
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            out.append((row, col))
        return out

    @staticmethod
    def _dominant_team(tracklet: Tracklet, player_by_id: dict[str, RosterPlayer]) -> str | None:
        if not tracklet.identity_posterior:
            return None
        best_id = max(tracklet.identity_posterior, key=tracklet.identity_posterior.get)
        player = player_by_id.get(best_id)
        return player.team_name if player else None

    def _resolve_conflicts(self, active: list[Tracklet], player_by_id: dict[str, RosterPlayer]) -> None:
        by_player: dict[str, list[Tracklet]] = {}
        for tracklet in active:
            if not tracklet.resolved_player_id:
                continue
            by_player.setdefault(tracklet.resolved_player_id, []).append(tracklet)

        for player_id, duplicates in by_player.items():
            if len(duplicates) <= 1:
                continue
            duplicates.sort(
                key=lambda item: item.identity_posterior.get(player_id, 0.0)
                * max(1.0, min(4.0, item.duration + 1.0)),
                reverse=True,
            )
            reserved = {duplicates[0].resolved_player_id}
            for tracklet in duplicates[1:]:
                replacement = self._best_available(tracklet, reserved, active, player_by_id)
                if replacement is None:
                    tracklet.resolved_player_id = None
                    tracklet.resolved_confidence = 0.0
                else:
                    tracklet.resolved_player_id = replacement
                    tracklet.resolved_confidence = float(tracklet.identity_posterior[replacement])
                    reserved.add(replacement)

    def _best_available(
        self,
        tracklet: Tracklet,
        reserved: set[str | None],
        active: list[Tracklet],
        player_by_id: dict[str, RosterPlayer],
    ) -> str | None:
        used = {item.resolved_player_id for item in active if item is not tracklet and item.resolved_player_id}
        used.update(reserved)
        if not tracklet.identity_posterior:
            return None
        current_team = None
        if tracklet.resolved_player_id and tracklet.resolved_player_id in player_by_id:
            current_team = player_by_id[tracklet.resolved_player_id].team_name
        for candidate_id, prob in sorted(tracklet.identity_posterior.items(), key=lambda item: item[1], reverse=True):
            if candidate_id in used:
                continue
            if current_team and player_by_id.get(candidate_id) and player_by_id[candidate_id].team_name != current_team:
                continue
            if prob < max(0.15, self.identity_confidence_threshold * 0.5):
                continue
            return candidate_id
        return None


def build_assigner(config: dict[str, Any], players: list[RosterPlayer], body_memory: BodyReIDMemory) -> WindowHungarianAssigner:
    assignment_config = config.get("assignment", {})
    return WindowHungarianAssigner(
        players=players,
        body_memory=body_memory,
        identity_confidence_threshold=float(assignment_config.get("identity_confidence_threshold", 0.4)),
        window_sec=float(assignment_config.get("window_sec", 1.0)),
        step_sec=float(assignment_config.get("step_sec", 0.5)),
    )
