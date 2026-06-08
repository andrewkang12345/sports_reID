from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from soccer_identity.utils.geometry import cosine_similarity, image_embedding
from soccer_identity.utils.schemas import Tracklet


def extract_body_embedding(crop: np.ndarray) -> list[float] | None:
    if crop.size == 0:
        return None
    emb = image_embedding(crop, size=(32, 64))
    return [float(v) for v in emb.tolist()]


def aggregate_body_embedding(tracklet: Tracklet) -> np.ndarray | None:
    embeddings = []
    weights = []
    for obs in tracklet.observations:
        if obs.body_embedding is None:
            continue
        embeddings.append(obs.body_embedding)
        weights.append(max(0.01, obs.crop_quality))
    if not embeddings:
        return None
    data = np.asarray(embeddings, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    emb = np.average(data, axis=0, weights=w).astype(np.float32)
    norm = float(np.linalg.norm(emb))
    if norm > 1e-8:
        emb = emb / norm
    return emb.astype(np.float32)


@dataclass
class BodyReIDMemory:
    enabled: bool = True
    player_memory: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.player_memory is None:
            self.player_memory = {}

    def similarity(self, tracklet: Tracklet, player_id: str) -> float:
        if not self.enabled or self.player_memory is None or player_id not in self.player_memory:
            return 0.0
        emb = aggregate_body_embedding(tracklet)
        if emb is None:
            return 0.0
        return float((cosine_similarity(emb, self.player_memory[player_id]) + 1.0) * 0.5)

    def update(self, tracklet: Tracklet, player_id: str, momentum: float = 0.8) -> None:
        if not self.enabled or self.player_memory is None:
            return
        emb = aggregate_body_embedding(tracklet)
        if emb is None:
            return
        if player_id in self.player_memory:
            mixed = momentum * self.player_memory[player_id] + (1.0 - momentum) * emb
            norm = float(np.linalg.norm(mixed))
            self.player_memory[player_id] = mixed / max(norm, 1e-8)
        else:
            self.player_memory[player_id] = emb


def body_reid_confidence(tracklet: Tracklet) -> float:
    qualities = [obs.crop_quality for obs in tracklet.observations if obs.body_embedding is not None]
    if not qualities:
        return 0.0
    duration_score = min(1.0, len(qualities) / 20.0)
    return float(0.4 * np.mean(qualities) + 0.6 * duration_score)


def build_body_reid_memory(config: dict[str, Any]) -> BodyReIDMemory:
    body_config = config.get("body_reid", {})
    return BodyReIDMemory(enabled=bool(body_config.get("enabled", True)))
