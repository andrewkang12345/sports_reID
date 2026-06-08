from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from soccer_identity.utils.geometry import cosine_similarity, crop_quality, image_embedding
from soccer_identity.utils.schemas import RosterPlayer, Tracklet


def extract_head_embedding(crop: np.ndarray) -> tuple[list[float] | None, float]:
    if crop.size == 0:
        return None, 0.0
    h, w = crop.shape[:2]
    if h < 24 or w < 12:
        return None, 0.0
    head = crop[: max(8, int(h * 0.34)), :]
    quality = crop_quality(head)
    if quality < 0.08:
        return None, quality
    emb = image_embedding(head, size=(32, 32))
    return [float(v) for v in emb.tolist()], float(quality)


@dataclass
class HeadshotMatcher:
    players: list[RosterPlayer]
    metadata_dir: Path
    enabled: bool = True
    roster_embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enabled:
            self._load_roster_headshots()

    def match_tracklet(self, tracklet: Tracklet) -> dict[str, float]:
        if not self.roster_embeddings:
            return {}
        embeddings = []
        weights = []
        for obs in tracklet.observations:
            if obs.head_embedding is None:
                continue
            embeddings.append(obs.head_embedding)
            weights.append(max(0.01, obs.head_quality))
        if not embeddings:
            return {}
        data = np.asarray(embeddings, dtype=np.float32)
        w = np.asarray(weights, dtype=np.float32)
        emb = np.average(data, axis=0, weights=w).astype(np.float32)
        out: dict[str, float] = {}
        for player in self.players:
            roster_emb = self.roster_embeddings.get(player.player_id)
            if roster_emb is None:
                continue
            out[player.player_id] = float((cosine_similarity(emb, roster_emb) + 1.0) * 0.5)
        return out

    def _load_roster_headshots(self) -> None:
        for player in self.players:
            if not player.headshot_path:
                continue
            path = Path(player.headshot_path)
            if not path.is_absolute():
                path = self.metadata_dir / path
            if not path.exists():
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            self.roster_embeddings[player.player_id] = image_embedding(image, size=(32, 32))


def build_headshot_matcher(config: dict[str, Any], players: list[RosterPlayer], metadata_path: str | Path) -> HeadshotMatcher:
    head_config = config.get("headshot_matcher", {})
    return HeadshotMatcher(
        players=players,
        metadata_dir=Path(metadata_path).resolve().parent,
        enabled=bool(head_config.get("enabled", True)),
    )
