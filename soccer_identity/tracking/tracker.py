from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from soccer_identity.utils.geometry import bbox_center, iou_xyxy, point_distance
from soccer_identity.utils.schemas import BBox, Detection

try:  # pragma: no cover - optional fast assignment path
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    last_frame_index: int
    last_timestamp: float
    hits: int = 1
    misses: int = 0
    confidence: float = 0.0


@dataclass
class TrackedDetection:
    track_id: str
    bbox: BBox
    confidence: float
    detection: Detection


class MultiObjectTracker:
    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        raise NotImplementedError

    def finish(self) -> None:
        return None


class BoxmotBoTSORTTracker(MultiObjectTracker):
    """boxmot's BoT-SORT with a real ReID embedding (OSNet by default, PRTReID-soccer if
    configured). The ReID embedding is the key improvement vs Ultralytics' built-in BoT-SORT
    (which reuses YOLO neck features and does not survive long occlusions). Real ReID means
    a player who's occluded for ~2-3s and re-detected gets the SAME track ID, eliminating
    the fragmentation that drives "label teleport" downstream.
    """

    def __init__(self, reid_weights: str, device: str = "cuda", **kwargs: Any) -> None:
        from boxmot import BoTSORT
        from pathlib import Path as _Path

        # Strip our own knobs from kwargs before passing to BoTSORT.
        knob_keys = {
            "backend", "tracker_config", "max_age_frames", "min_iou",
            "max_center_distance", "iou_weight", "center_weight",
        }
        botsort_kwargs = {k: v for k, v in kwargs.items() if k not in knob_keys}
        self.tracker = BoTSORT(
            model_weights=_Path(reid_weights),
            device=device,
            fp16=False,
            with_reid=True,
            **botsort_kwargs,
        )

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        if frame is None:
            # boxmot needs the raw frame to crop and embed; without it we can't track.
            return []
        if not detections:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            rows = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox.xyxy
                rows.append([x1, y1, x2, y2, det.confidence, 0.0])  # class_id 0 = person
            dets = np.asarray(rows, dtype=np.float32)
        # boxmot returns [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        out = self.tracker.update(dets, frame)
        tracked: list[TrackedDetection] = []
        for row in out:
            x1, y1, x2, y2, tid, conf, _cls, det_idx = row.tolist()
            di = int(det_idx)
            detection = detections[di] if 0 <= di < len(detections) else None
            if detection is None:
                continue
            bbox = BBox(float(x1), float(y1), float(x2), float(y2)).clipped(int(frame.shape[1]), int(frame.shape[0]))
            tracked.append(TrackedDetection(str(int(tid)), bbox, float(conf), detection))
        return tracked


class DetectorIdTracker(MultiObjectTracker):
    def __init__(self, fallback: MultiObjectTracker | None = None) -> None:
        self.fallback = fallback or SimpleIoUTracker()

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        if detections and all(det.attributes.get("track_id") is not None for det in detections):
            return [
                TrackedDetection(
                    track_id=str(det.attributes["track_id"]),
                    bbox=det.bbox,
                    confidence=det.confidence,
                    detection=det,
                )
                for det in detections
            ]
        return self.fallback.update(detections, frame_index, timestamp, frame)


@dataclass
class SimpleIoUTracker(MultiObjectTracker):
    max_age_frames: int = 12
    min_iou: float = 0.05
    max_center_distance: float = 120.0
    iou_weight: float = 0.65
    center_weight: float = 0.35
    next_track_id: int = 1
    tracks: dict[int, TrackState] = field(default_factory=dict)

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        active_ids = list(self.tracks.keys())
        if active_ids and detections:
            cost = np.ones((len(active_ids), len(detections)), dtype=np.float32) * 1e3
            for row, track_id in enumerate(active_ids):
                track = self.tracks[track_id]
                for col, detection in enumerate(detections):
                    iou = iou_xyxy(track.bbox.xyxy, detection.bbox.xyxy)
                    dist = point_distance(bbox_center(track.bbox.xyxy), bbox_center(detection.bbox.xyxy))
                    if iou < self.min_iou and dist > self.max_center_distance:
                        continue
                    norm_dist = min(1.0, dist / max(self.max_center_distance, 1.0))
                    cost[row, col] = self.iou_weight * (1.0 - iou) + self.center_weight * norm_dist
            matches, unmatched_tracks, unmatched_detections = self._assign(cost, active_ids, detections)
        else:
            matches = []
            unmatched_tracks = set(active_ids)
            unmatched_detections = set(range(len(detections)))

        tracked: list[TrackedDetection] = []
        for track_id, detection_index in matches:
            detection = detections[detection_index]
            track = self.tracks[track_id]
            track.bbox = detection.bbox
            track.last_frame_index = frame_index
            track.last_timestamp = timestamp
            track.hits += 1
            track.misses = 0
            track.confidence = 0.8 * track.confidence + 0.2 * detection.confidence if track.confidence else detection.confidence
            tracked.append(TrackedDetection(str(track_id), detection.bbox, detection.confidence, detection))

        for track_id in unmatched_tracks:
            if track_id in self.tracks:
                self.tracks[track_id].misses += 1

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                last_frame_index=frame_index,
                last_timestamp=timestamp,
                confidence=detection.confidence,
            )
            tracked.append(TrackedDetection(str(track_id), detection.bbox, detection.confidence, detection))

        stale = [track_id for track_id, track in self.tracks.items() if track.misses > self.max_age_frames]
        for track_id in stale:
            del self.tracks[track_id]

        tracked.sort(key=lambda item: int(item.track_id))
        return tracked

    def _assign(
        self,
        cost: np.ndarray,
        active_ids: list[int],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        unmatched_tracks = set(active_ids)
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []
        if cost.size == 0:
            return matches, unmatched_tracks, unmatched_detections
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            assignments = zip(rows.tolist(), cols.tolist())
        else:  # pragma: no cover
            flat = sorted((float(cost[r, c]), r, c) for r in range(cost.shape[0]) for c in range(cost.shape[1]))
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            assignments_list = []
            for value, row, col in flat:
                if row in used_rows or col in used_cols:
                    continue
                used_rows.add(row)
                used_cols.add(col)
                assignments_list.append((row, col))
            assignments = assignments_list

        for row, col in assignments:
            if float(cost[row, col]) >= 0.95:
                continue
            track_id = active_ids[row]
            matches.append((track_id, col))
            unmatched_tracks.discard(track_id)
            unmatched_detections.discard(col)
        return matches, unmatched_tracks, unmatched_detections


def build_tracker(config: dict[str, Any]) -> MultiObjectTracker:
    tracker_config = config.get("tracker", {})
    backend = str(tracker_config.get("backend", "simple_iou")).lower()
    if backend not in {"simple_iou", "detector", "detector_ids", "bytetrack", "botsort", "strongsort", "deepsort", "sportsmot", "mixsort", "sportsmot_mixsort", "boxmot_botsort", "mixsort_mixformer"}:
        raise ValueError(f"Unsupported tracker backend: {backend}")
    if backend == "mixsort_mixformer":
        from soccer_identity.tracking.mixsort_wrapper import MixSortTracker
        return MixSortTracker(
            mixformer_weights=str(tracker_config.get("mixformer_weights", "models/mixsort/MixFormer_soccernet_train.pth.tar")),
            device=str(tracker_config.get("device", "cuda:0")),
            track_thresh=float(tracker_config.get("track_thresh", 0.5)),
            track_buffer=int(tracker_config.get("track_buffer", 60)),
            match_thresh=float(tracker_config.get("match_thresh", 0.8)),
            alpha=float(tracker_config.get("alpha", 0.6)),
            radius=int(tracker_config.get("radius", 0)),
            iou_thresh=float(tracker_config.get("iou_thresh", 0.3)),
            script=str(tracker_config.get("script", "mixformer_deit")),
            config=str(tracker_config.get("mixformer_config", "soccernet")),
            frame_rate=int(tracker_config.get("frame_rate", 30)),
        )
    if backend == "boxmot_botsort":
        reid_weights = str(tracker_config.get("reid_weights", "osnet_x0_25_msmt17.pt"))
        device = str(tracker_config.get("device", "cuda:0"))
        return BoxmotBoTSORTTracker(
            reid_weights=reid_weights,
            device=device,
            track_high_thresh=float(tracker_config.get("track_high_thresh", 0.25)),
            track_low_thresh=float(tracker_config.get("track_low_thresh", 0.10)),
            new_track_thresh=float(tracker_config.get("new_track_thresh", 0.25)),
            track_buffer=int(tracker_config.get("track_buffer", 90)),
            match_thresh=float(tracker_config.get("match_thresh", 0.75)),
            proximity_thresh=float(tracker_config.get("proximity_thresh", 0.5)),
            appearance_thresh=float(tracker_config.get("appearance_thresh", 0.25)),
            cmc_method=str(tracker_config.get("cmc_method", "sof")),
            frame_rate=int(tracker_config.get("frame_rate", 30)),
            fuse_first_associate=bool(tracker_config.get("fuse_first_associate", True)),
        )
    simple_tracker = SimpleIoUTracker(
        max_age_frames=int(tracker_config.get("max_age_frames", 12)),
        min_iou=float(tracker_config.get("min_iou", 0.05)),
        max_center_distance=float(tracker_config.get("max_center_distance", 120.0)),
        iou_weight=float(tracker_config.get("iou_weight", 0.65)),
        center_weight=float(tracker_config.get("center_weight", 0.35)),
    )
    if backend in {"sportsmot", "mixsort", "sportsmot_mixsort"}:
        from soccer_identity.tracking.sportsmot_adapter import build_sportsmot_adapter

        return build_sportsmot_adapter(config, simple_tracker=simple_tracker)
    if backend in {"detector", "detector_ids", "bytetrack", "botsort"} and bool(config.get("detector", {}).get("use_tracking", False)):
        return DetectorIdTracker(fallback=simple_tracker)
    if backend != "simple_iou":
        # Adapter hook: wrap ByteTrack/BoT-SORT/etc. here in production.
        # The fallback remains deterministic and dependency-light for this prototype.
        pass
    return simple_tracker
