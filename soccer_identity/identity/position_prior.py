from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from soccer_identity.utils.schemas import RosterPlayer, Tracklet


@dataclass
class PositionPrior:
    enabled: bool = True
    default_prior: float = 0.5
    strength: float = 0.12

    def score(self, tracklet: Tracklet, player: RosterPlayer) -> float:
        if not self.enabled or not player.nominal_position:
            return self.default_prior
        if not tracklet.observations:
            return self.default_prior
        # Without pitch calibration, position is intentionally weak. We only use
        # stable image trajectory cues to avoid making role a hard constraint.
        centers = np.asarray([obs.bbox.center for obs in tracklet.observations], dtype=np.float32)
        avg_y = float(np.mean(centers[:, 1]))
        span_y = float(np.ptp(centers[:, 1])) if len(centers) > 1 else 0.0
        role = str(player.nominal_position).upper()
        modifier = 0.0
        if role.startswith("GK"):
            modifier = -0.04 if span_y > 35 else 0.03
        elif role.startswith("DF"):
            modifier = 0.01 if span_y < 90 else -0.01
        elif role.startswith("MF"):
            modifier = 0.02 if span_y >= 15 else 0.0
        elif role.startswith("FW") or role.startswith("ST"):
            modifier = 0.01 if avg_y > 0 else 0.0
        return float(max(0.05, min(0.95, self.default_prior + self.strength * modifier)))


def build_position_prior(config: dict[str, Any]) -> PositionPrior:
    prior_config = config.get("position_prior", {})
    return PositionPrior(
        enabled=bool(prior_config.get("enabled", True)),
        default_prior=float(prior_config.get("default_prior", 0.5)),
        strength=float(prior_config.get("strength", 0.12)),
    )
