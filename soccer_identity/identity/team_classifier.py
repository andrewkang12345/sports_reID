from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from soccer_identity.utils.geometry import crop_xyxy, dominant_non_green_rgb, dominant_non_green_rgb_region, parse_hex_color, softmax
from soccer_identity.utils.schemas import BBox, Tracklet


def _kit_color_distance(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """Kit-color distance in LAB space with the L (lightness) axis downweighted.

    Real broadcast jerseys vary in apparent brightness from stadium lighting, so a pure-white
    reference (255,255,255) vs an observed off-white (~200,200,200) looks far in RGB but is
    perceptually close. LAB-with-reduced-L matches what a human would call "the same kit".
    """
    arr_a = np.asarray(rgb_a, dtype=np.float32).reshape(1, 1, 3)
    arr_b = np.asarray(rgb_b, dtype=np.float32).reshape(1, 1, 3)
    lab_a = cv2.cvtColor(np.clip(arr_a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    lab_b = cv2.cvtColor(np.clip(arr_b, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dl = (lab_a[0] - lab_b[0]) * 0.4
    da = lab_a[1] - lab_b[1]
    db = lab_a[2] - lab_b[2]
    return float(np.sqrt(dl * dl + da * da + db * db) / 255.0)


def extract_team_color(frame: np.ndarray, bbox: BBox) -> tuple[list[float] | None, float]:
    crop = crop_xyxy(frame, bbox.xyxy)
    rgb, quality = dominant_non_green_rgb(crop)
    if rgb is None:
        return None, 0.0
    return [float(v) for v in rgb.tolist()], float(quality)


def extract_kit_colors(frame: np.ndarray, bbox: BBox) -> tuple[list[float] | None, float, list[float] | None, float]:
    crop = crop_xyxy(frame, bbox.xyxy)
    shirt_rgb, shirt_quality = dominant_non_green_rgb_region(crop, 0.18, 0.58)
    shorts_rgb, shorts_quality = dominant_non_green_rgb_region(crop, 0.54, 0.86, 0.18, 0.82)
    shirt = [float(v) for v in shirt_rgb.tolist()] if shirt_rgb is not None else None
    shorts = [float(v) for v in shorts_rgb.tolist()] if shorts_rgb is not None else None
    return shirt, float(shirt_quality), shorts, float(shorts_quality)


@dataclass
class TeamColorClassifier:
    team_names: list[str]
    metadata: dict[str, Any]
    temperature: float = 0.16
    unfit_confidence_cap: float = 0.62
    unknown_color_distance: float = 0.48
    team_color_refs: dict[str, np.ndarray] = field(default_factory=dict)
    team_kit_refs: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    cluster_centers: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.team_kit_refs = self._parse_team_kits(self.metadata.get("team_colors", {}))
        self.team_color_refs = {team: refs["shirt"] for team, refs in self.team_kit_refs.items() if "shirt" in refs}

    def fit(self, tracklets: list[Tracklet]) -> None:
        if self.team_color_refs or len(self.team_names) < 2:
            return
        colors = []
        for tracklet in tracklets:
            color = self.tracklet_color(tracklet)
            if color is not None:
                colors.append(color)
        if len(colors) < 2:
            return
        data = np.asarray(colors, dtype=np.float32)
        centers = np.asarray([data[np.argmin(data[:, 0])], data[np.argmax(data[:, 0])]], dtype=np.float32)
        for _ in range(12):
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            for idx in range(2):
                if np.any(labels == idx):
                    centers[idx] = data[labels == idx].mean(axis=0)
        # Deterministic fallback mapping. Confidence is capped because this is not semantically anchored.
        for team_name, center in zip(self.team_names[:2], centers):
            self.cluster_centers[team_name] = center.astype(np.float32)

    def predict_tracklet(self, tracklet: Tracklet) -> dict[str, float]:
        if not self.team_names:
            return {}
        color = self.tracklet_color(tracklet)
        if color is None:
            return {team: 1.0 / len(self.team_names) for team in self.team_names}
        refs = self.team_color_refs or self.cluster_centers
        if not refs:
            return {team: 1.0 / len(self.team_names) for team in self.team_names}
        distances = []
        teams = []
        for team in self.team_names:
            team_distance = self._team_shirt_distance(tracklet, team)
            if team_distance is None:
                ref = refs.get(team)
                if ref is None:
                    continue
                team_distance = _kit_color_distance(color, ref)
            if team_distance is None:
                continue
            distances.append(team_distance)
            teams.append(team)
        if not teams:
            return {team: 1.0 / len(self.team_names) for team in self.team_names}
        if self.team_color_refs and min(distances) > self.unknown_color_distance:
            return {team: 0.02 for team in self.team_names}
        probs = softmax([-d for d in distances], temperature=self.temperature)
        out = {team: 1e-4 for team in self.team_names}
        for team, prob in zip(teams, probs):
            out[team] = float(prob)
        total = sum(out.values())
        out = {team: value / total for team, value in out.items()}
        if not self.team_color_refs:
            winner = max(out, key=out.get)
            capped = min(out[winner], self.unfit_confidence_cap)
            remainder = 1.0 - capped
            other_teams = [team for team in out if team != winner]
            out = {team: remainder / max(1, len(other_teams)) for team in out}
            out[winner] = capped
        return out

    def team_fit_score(self, tracklet: Tracklet) -> float:
        refs = self.team_kit_refs or {team: {"shirt": ref} for team, ref in self.cluster_centers.items()}
        if not refs:
            return 0.5
        fits = [self._team_kit_fit(tracklet, team) for team in refs]
        fits = [fit for fit in fits if fit is not None]
        if not fits:
            return 0.5
        return float(max(fits))

    def tracklet_color(self, tracklet: Tracklet) -> np.ndarray | None:
        colors = []
        weights = []
        for obs in tracklet.observations:
            if obs.team_color_rgb is None:
                continue
            colors.append(obs.team_color_rgb)
            weights.append(max(0.05, obs.team_color_quality))
        if not colors:
            return None
        data = np.asarray(colors, dtype=np.float32)
        w = np.asarray(weights, dtype=np.float32)
        return np.average(data, axis=0, weights=w).astype(np.float32)

    def tracklet_shorts_color(self, tracklet: Tracklet) -> tuple[np.ndarray | None, float]:
        colors = []
        weights = []
        for obs in tracklet.observations:
            if obs.shorts_color_rgb is None:
                continue
            colors.append(obs.shorts_color_rgb)
            weights.append(max(0.05, obs.shorts_color_quality))
        if not colors:
            return None, 0.0
        data = np.asarray(colors, dtype=np.float32)
        w = np.asarray(weights, dtype=np.float32)
        return np.average(data, axis=0, weights=w).astype(np.float32), float(np.mean(weights))

    def _team_shirt_distance(self, tracklet: Tracklet, team: str) -> float | None:
        color = self.tracklet_color(tracklet)
        ref = self.team_kit_refs.get(team, {}).get("shirt")
        if color is None or ref is None:
            return None
        return _kit_color_distance(color, ref)

    def _team_kit_fit(self, tracklet: Tracklet, team: str) -> float | None:
        refs = self.team_kit_refs.get(team)
        if not refs:
            return None
        shirt_color = self.tracklet_color(tracklet)
        if shirt_color is None or "shirt" not in refs:
            return None
        shirt_distance = _kit_color_distance(shirt_color, refs["shirt"])
        shirt_fit = max(0.0, min(1.0, 1.0 - shirt_distance / max(self.unknown_color_distance, 1e-6)))
        shorts_ref = refs.get("shorts")
        shorts_color, shorts_quality = self.tracklet_shorts_color(tracklet)
        if shorts_ref is None or shorts_color is None or shorts_quality < 0.08:
            return shirt_fit
        shorts_distance = _kit_color_distance(shorts_color, shorts_ref)
        shorts_fit = max(0.0, min(1.0, 1.0 - shorts_distance / max(self.unknown_color_distance, 1e-6)))
        combined = 0.58 * shirt_fit + 0.42 * shorts_fit
        if shirt_fit >= 0.70 and shorts_fit < 0.25:
            combined = min(combined, 0.42)
        return float(combined)

    @staticmethod
    def _parse_team_kits(raw: Any) -> dict[str, dict[str, np.ndarray]]:
        refs: dict[str, dict[str, np.ndarray]] = {}
        if not isinstance(raw, dict):
            return refs
        for team, value in raw.items():
            try:
                if isinstance(value, str):
                    refs[str(team)] = {"shirt": parse_hex_color(value)}
                elif isinstance(value, dict):
                    shirt_value = value.get("shirt") or value.get("primary") or value.get("home") or value.get("color")
                    shorts_value = value.get("shorts") or value.get("secondary")
                    team_refs: dict[str, np.ndarray] = {}
                    if shirt_value:
                        team_refs["shirt"] = parse_hex_color(str(shirt_value))
                    if shorts_value:
                        team_refs["shorts"] = parse_hex_color(str(shorts_value))
                    if team_refs:
                        refs[str(team)] = team_refs
            except ValueError:
                continue
        return refs


def build_team_classifier(config: dict[str, Any], metadata: dict[str, Any]) -> TeamColorClassifier:
    teams = [metadata.get("home_team"), metadata.get("away_team")]
    team_names = [str(team) for team in teams if team]
    team_config = config.get("team_classifier", {})
    return TeamColorClassifier(
        team_names=team_names,
        metadata=metadata,
        temperature=float(team_config.get("temperature", 0.16)),
        unfit_confidence_cap=float(team_config.get("unfit_confidence_cap", 0.62)),
        unknown_color_distance=float(team_config.get("unknown_color_distance", 0.48)),
    )
