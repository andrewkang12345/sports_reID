from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from soccer_identity.identity.body_reid import BodyReIDMemory, body_reid_confidence
from soccer_identity.identity.headshot_matcher import HeadshotMatcher
from soccer_identity.identity.jersey_ocr import aggregate_jersey_probs
from soccer_identity.identity.position_prior import PositionPrior
from soccer_identity.identity.team_classifier import TeamColorClassifier
from soccer_identity.utils.geometry import safe_log, softmax
from soccer_identity.utils.schemas import RosterPlayer, Tracklet


@dataclass
class FusionWeights:
    team: float = 2.0
    jersey: float = 3.2
    headshot: float = 1.0
    body: float = 0.6
    position: float = 0.35
    track: float = 0.25


@dataclass
class TrackletEvidence:
    team_probs: dict[str, float]
    jersey_probs: dict[str, float]
    jersey_quality: float
    headshot_sims: dict[str, float]
    tracking_confidence: float
    body_reid_confidence: float
    player_likelihood: float = 1.0


class RosterIdentityResolver:
    def __init__(
        self,
        players: list[RosterPlayer],
        team_classifier: TeamColorClassifier,
        headshot_matcher: HeadshotMatcher,
        body_memory: BodyReIDMemory,
        position_prior: PositionPrior,
        config: dict[str, Any],
    ) -> None:
        self.players = players
        self.player_by_id = {player.player_id: player for player in players}
        self.team_classifier = team_classifier
        self.headshot_matcher = headshot_matcher
        self.body_memory = body_memory
        self.position_prior = position_prior
        fusion_config = config.get("fusion", {})
        weights = fusion_config.get("weights", {})
        self.weights = FusionWeights(
            team=float(weights.get("team", 2.0)),
            jersey=float(weights.get("jersey", 3.2)),
            headshot=float(weights.get("headshot", 1.0)),
            body=float(weights.get("body", 0.6)),
            position=float(weights.get("position", 0.35)),
            track=float(weights.get("track", 0.25)),
        )
        self.temperature = float(fusion_config.get("temperature", 1.0))
        self.jersey_missing_neutral = float(fusion_config.get("jersey_missing_neutral", 0.72))
        self.team_missing_neutral = float(fusion_config.get("team_missing_neutral", 0.5))

    def resolve(self, tracklets: list[Tracklet]) -> dict[str, TrackletEvidence]:
        self.team_classifier.fit(tracklets)
        debug_evidence: dict[str, TrackletEvidence] = {}
        for tracklet in tracklets:
            evidence = self._collect_evidence(tracklet)
            scores = self._score_tracklet(tracklet, evidence)
            if scores:
                player_ids = list(scores.keys())
                probs_arr = softmax(list(scores.values()), temperature=self.temperature)
                posterior = {player_id: float(prob) for player_id, prob in zip(player_ids, probs_arr)}
            else:
                posterior = {}
            tracklet.identity_scores = {key: float(value) for key, value in scores.items()}
            tracklet.identity_posterior = posterior
            if posterior:
                best_id = max(posterior, key=posterior.get)
                best_player = self.player_by_id[best_id]
                jersey_conf = evidence.jersey_probs.get(best_player.jersey_number or "", 0.0) if evidence.jersey_probs else 0.0
                head_conf = evidence.headshot_sims.get(best_id, 0.0)
                tracklet.resolved_player_id = best_id
                tracklet.resolved_confidence = float(posterior[best_id])
                tracklet.evidence = {
                    "team_confidence": float(evidence.team_probs.get(best_player.team_name, 0.0)),
                    "jersey_confidence": float(jersey_conf),
                    "headshot_confidence": float(head_conf),
                    "body_reid_confidence": float(evidence.body_reid_confidence),
                    "tracking_confidence": float(evidence.tracking_confidence),
                    "position_prior": float(self.position_prior.score(tracklet, best_player)),
                }
            debug_evidence[tracklet.track_id] = evidence
        return debug_evidence

    def _collect_evidence(self, tracklet: Tracklet) -> TrackletEvidence:
        jersey_probs, jersey_quality = aggregate_jersey_probs(tracklet)
        tracking_conf = min(0.99, 0.35 + 0.35 * min(1.0, len(tracklet.observations) / 18.0) + 0.30 * tracklet.average_detection_confidence())
        return TrackletEvidence(
            team_probs=self.team_classifier.predict_tracklet(tracklet),
            jersey_probs=jersey_probs,
            jersey_quality=jersey_quality,
            headshot_sims=self.headshot_matcher.match_tracklet(tracklet),
            tracking_confidence=float(tracking_conf),
            body_reid_confidence=body_reid_confidence(tracklet),
            player_likelihood=float(self.team_classifier.team_fit_score(tracklet)),
        )

    def _score_tracklet(self, tracklet: Tracklet, evidence: TrackletEvidence) -> dict[str, float]:
        scores: dict[str, float] = {}
        for player in self.players:
            team_prob = evidence.team_probs.get(player.team_name, self.team_missing_neutral)
            score = self.weights.team * safe_log(team_prob)

            if evidence.jersey_probs and player.jersey_number:
                jersey_prob = evidence.jersey_probs.get(player.jersey_number, 1e-4)
                # Quality gates jersey impact. Reliable jersey observations should dominate.
                score += self.weights.jersey * max(0.15, evidence.jersey_quality) * safe_log(jersey_prob)
            elif evidence.jersey_probs:
                score += self.weights.jersey * max(0.15, evidence.jersey_quality) * safe_log(1e-4)
            else:
                score += self.weights.jersey * 0.05 * safe_log(self.jersey_missing_neutral)

            headshot_sim = evidence.headshot_sims.get(player.player_id)
            if headshot_sim is not None:
                score += self.weights.headshot * (2.0 * headshot_sim - 1.0)

            body_sim = self.body_memory.similarity(tracklet, player.player_id)
            if body_sim > 0:
                score += self.weights.body * (2.0 * body_sim - 1.0)

            pos_prior = self.position_prior.score(tracklet, player)
            score += self.weights.position * safe_log(pos_prior)
            score += self.weights.track * safe_log(evidence.tracking_confidence)
            scores[player.player_id] = float(score)
        return scores


def build_identity_resolver(
    config: dict[str, Any],
    players: list[RosterPlayer],
    team_classifier: TeamColorClassifier,
    headshot_matcher: HeadshotMatcher,
    body_memory: BodyReIDMemory,
    position_prior: PositionPrior,
) -> RosterIdentityResolver:
    return RosterIdentityResolver(
        players=players,
        team_classifier=team_classifier,
        headshot_matcher=headshot_matcher,
        body_memory=body_memory,
        position_prior=position_prior,
        config=config,
    )
