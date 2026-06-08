from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from soccer_identity.utils.geometry import bbox_area, bbox_bottom_center, bbox_center


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.xyxy)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return bbox_bottom_center(self.xyxy)

    @property
    def area(self) -> float:
        return bbox_area(self.xyxy)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def clipped(self, width: int, height: int) -> "BBox":
        return BBox(
            max(0.0, min(float(width - 1), self.x1)),
            max(0.0, min(float(height - 1), self.y1)),
            max(0.0, min(float(width - 1), self.x2)),
            max(0.0, min(float(height - 1), self.y2)),
        )

    def to_list(self) -> list[float]:
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2)]


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_name: str = "player"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RosterPlayer:
    player_id: str
    player_name: str
    team_name: str
    jersey_number: str | None = None
    nominal_position: str | None = None
    headshot_path: str | None = None
    headshot_url: str | None = None
    headshot_embedding: list[float] | None = None

    def display_name(self) -> str:
        if self.jersey_number:
            return f"{self.player_name} #{self.jersey_number}"
        return self.player_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team_name": self.team_name,
            "jersey_number": self.jersey_number,
            "nominal_position": self.nominal_position,
            "headshot_path": self.headshot_path,
            "headshot_url": self.headshot_url,
        }


@dataclass
class TrackObservation:
    frame_index: int
    timestamp: float
    bbox: BBox
    detection_confidence: float
    team_color_rgb: list[float] | None = None
    team_color_quality: float = 0.0
    shorts_color_rgb: list[float] | None = None
    shorts_color_quality: float = 0.0
    jersey_probs: dict[str, float] = field(default_factory=dict)
    jersey_quality: float = 0.0
    head_embedding: list[float] | None = None
    head_quality: float = 0.0
    body_embedding: list[float] | None = None
    appearance_embedding: list[float] | None = None  # real ReID embedding (OSNet/CLIP-ReID)
    crop_quality: float = 0.0
    occlusion_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp": float(self.timestamp),
            "bbox": self.bbox.to_list(),
            "detection_confidence": float(self.detection_confidence),
            "team_color_rgb": self.team_color_rgb,
            "team_color_quality": float(self.team_color_quality),
            "shorts_color_rgb": self.shorts_color_rgb,
            "shorts_color_quality": float(self.shorts_color_quality),
            "jersey_probs": self.jersey_probs,
            "jersey_quality": float(self.jersey_quality),
            "head_quality": float(self.head_quality),
            "appearance_embedding": self.appearance_embedding,
            "crop_quality": float(self.crop_quality),
            "occlusion_score": float(self.occlusion_score),
        }


@dataclass
class Tracklet:
    track_id: str
    observations: list[TrackObservation] = field(default_factory=list)
    identity_scores: dict[str, float] = field(default_factory=dict)
    identity_posterior: dict[str, float] = field(default_factory=dict)
    resolved_player_id: str | None = None
    resolved_confidence: float = 0.0
    evidence: dict[str, float] = field(default_factory=dict)
    is_player: bool = True
    player_likelihood: float = 1.0

    @property
    def start_time(self) -> float:
        return min((obs.timestamp for obs in self.observations), default=0.0)

    @property
    def end_time(self) -> float:
        return max((obs.timestamp for obs in self.observations), default=0.0)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def frame_indices(self) -> list[int]:
        return [obs.frame_index for obs in self.observations]

    def average_bbox_area(self) -> float:
        if not self.observations:
            return 0.0
        return float(np.mean([obs.bbox.area for obs in self.observations]))

    def average_detection_confidence(self) -> float:
        if not self.observations:
            return 0.0
        return float(np.mean([obs.detection_confidence for obs in self.observations]))

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "num_observations": len(self.observations),
            "resolved_player_id": self.resolved_player_id,
            "resolved_confidence": self.resolved_confidence,
            "is_player": self.is_player,
            "player_likelihood": self.player_likelihood,
            "evidence": self.evidence,
            "observations": [obs.to_dict() for obs in self.observations],
        }


def normalize_jersey_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def make_player_id(team_name: str, player_name: str, jersey_number: str | None) -> str:
    jersey = jersey_number or "NA"
    return f"{team_name}|{jersey}|{player_name}".replace("/", "_")


def load_metadata(path: str | Path) -> tuple[dict[str, Any], list[RosterPlayer]]:
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    rosters = metadata.get("rosters", {})
    players: list[RosterPlayer] = []
    for team_name, entries in rosters.items():
        for entry in entries:
            jersey = normalize_jersey_number(entry.get("jersey_number"))
            position = entry.get("nominal_position", entry.get("position"))
            name = str(entry.get("player_name", "Unknown")).strip()
            player_id = entry.get("player_id") or make_player_id(team_name, name, jersey)
            players.append(
                RosterPlayer(
                    player_id=player_id,
                    player_name=name,
                    team_name=entry.get("team_name", team_name),
                    jersey_number=jersey,
                    nominal_position=position,
                    headshot_path=entry.get("headshot_path"),
                    headshot_url=entry.get("headshot_url"),
                )
            )
    return metadata, players


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
