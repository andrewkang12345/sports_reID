from __future__ import annotations

from dataclasses import dataclass, field

from soccer_identity.utils.schemas import BBox, TrackObservation, Tracklet


@dataclass
class TrackletBuilder:
    min_observations: int = 2
    tracklets: dict[str, Tracklet] = field(default_factory=dict)

    def add_observation(
        self,
        track_id: str,
        frame_index: int,
        timestamp: float,
        bbox: BBox,
        detection_confidence: float,
        team_color_rgb: list[float] | None = None,
        team_color_quality: float = 0.0,
        shorts_color_rgb: list[float] | None = None,
        shorts_color_quality: float = 0.0,
        jersey_probs: dict[str, float] | None = None,
        jersey_quality: float = 0.0,
        head_embedding: list[float] | None = None,
        head_quality: float = 0.0,
        body_embedding: list[float] | None = None,
        appearance_embedding: list[float] | None = None,
        crop_quality: float = 0.0,
        occlusion_score: float = 0.0,
    ) -> None:
        tracklet = self.tracklets.setdefault(track_id, Tracklet(track_id=track_id))
        tracklet.observations.append(
            TrackObservation(
                frame_index=frame_index,
                timestamp=timestamp,
                bbox=bbox,
                detection_confidence=detection_confidence,
                team_color_rgb=team_color_rgb,
                team_color_quality=team_color_quality,
                shorts_color_rgb=shorts_color_rgb,
                shorts_color_quality=shorts_color_quality,
                jersey_probs=jersey_probs or {},
                jersey_quality=jersey_quality,
                head_embedding=head_embedding,
                head_quality=head_quality,
                body_embedding=body_embedding,
                appearance_embedding=appearance_embedding,
                crop_quality=crop_quality,
                occlusion_score=occlusion_score,
            )
        )

    def finalize(self) -> list[Tracklet]:
        tracklets = [
            tracklet
            for tracklet in self.tracklets.values()
            if len(tracklet.observations) >= self.min_observations
        ]
        for tracklet in tracklets:
            tracklet.observations.sort(key=lambda obs: obs.frame_index)
        tracklets.sort(key=lambda item: (item.start_time, int(item.track_id) if item.track_id.isdigit() else item.track_id))
        return tracklets
