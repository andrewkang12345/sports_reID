from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soccer_identity.tracking.tracker import MultiObjectTracker, SimpleIoUTracker


@dataclass
class SportsMOTMixSortConfig:
    repo_path: str | None = None
    weights_path: str | None = None
    device: str = "cuda"
    fallback_to_simple: bool = True


class SportsMOTMixSortAdapter(MultiObjectTracker):
    """Adapter placeholder for SportsMOT/MixSort-style production tracking.

    SportsMOT is distributed as a dataset/codebase rather than a pip package.
    This adapter validates the external checkout and intentionally falls back
    to the internal online tracker unless a project-specific wrapper is added.
    """

    def __init__(self, config: SportsMOTMixSortConfig, simple_tracker: SimpleIoUTracker | None = None) -> None:
        self.config = config
        self.repo_path = Path(config.repo_path).expanduser() if config.repo_path else None
        self.weights_path = Path(config.weights_path).expanduser() if config.weights_path else None
        self.simple_tracker = simple_tracker or SimpleIoUTracker()
        self.available = self._is_available()
        if not self.available and not config.fallback_to_simple:
            raise FileNotFoundError(
                "SportsMOT/MixSort backend requested, but repo_path/weights_path are not available. "
                "Set tracker.sportsmot.fallback_to_simple=true for local smoke tests."
            )

    def update(self, detections: list[Any], frame_index: int, timestamp: float) -> list[Any]:
        if not self.available:
            return self.simple_tracker.update(detections, frame_index, timestamp)
        # Production integration point:
        # 1. Convert detector boxes to the MixSort expected detection format.
        # 2. Call the external tracker state update.
        # 3. Convert resulting tracks back to TrackedDetection.
        #
        # SportsMOT's public code layout has changed over time, so this repo
        # keeps the adapter boundary stable and avoids hard-coding a brittle
        # import path here.
        raise NotImplementedError(
            "SportsMOT/MixSort repo is present, but a local wrapper for its inference API "
            "has not been configured. Use BoT-SORT/ByteTrack or implement this adapter "
            "against your checked-out SportsMOT code."
        )

    def _is_available(self) -> bool:
        if self.repo_path is None or not self.repo_path.exists():
            return False
        if self.weights_path is not None and not self.weights_path.exists():
            return False
        return True


def build_sportsmot_adapter(config: dict[str, Any], simple_tracker: SimpleIoUTracker | None = None) -> SportsMOTMixSortAdapter:
    tracker_config = config.get("tracker", {})
    sportsmot_config = tracker_config.get("sportsmot", {})
    return SportsMOTMixSortAdapter(
        SportsMOTMixSortConfig(
            repo_path=sportsmot_config.get("repo_path"),
            weights_path=sportsmot_config.get("weights_path"),
            device=str(sportsmot_config.get("device", "cuda")),
            fallback_to_simple=bool(sportsmot_config.get("fallback_to_simple", True)),
        ),
        simple_tracker=simple_tracker,
    )
