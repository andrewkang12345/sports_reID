from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from soccer_identity.utils.geometry import bbox_area, iou_xyxy
from soccer_identity.utils.schemas import BBox, Detection


class PlayerDetector:
    """Interface for player detectors."""

    def detect(self, frame: np.ndarray, frame_index: int, timestamp: float) -> list[Detection]:
        raise NotImplementedError


@dataclass
class OpenCVPlayerDetector(PlayerDetector):
    min_area: int = 300
    max_area_ratio: float = 0.08
    min_height: int = 22
    min_aspect: float = 0.22
    max_aspect: float = 1.25
    confidence_floor: float = 0.35

    def detect(self, frame: np.ndarray, frame_index: int, timestamp: float) -> list[Detection]:
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        green_field = (hue >= 32) & (hue <= 92) & (sat >= 35) & (val >= 35)
        dark_or_colored = ((sat >= 35) & (val >= 35)) | ((val < 85) & (sat > 15))
        mask = (dark_or_colored & ~green_field).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        frame_area = float(width * height)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            if area < self.min_area or area > frame_area * self.max_area_ratio:
                continue
            if h < self.min_height:
                continue
            aspect = w / max(1.0, float(h))
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            if y < 2 or y + h > height - 2:
                continue
            bbox = BBox(float(x), float(y), float(x + w), float(y + h)).clipped(width, height)
            fill_ratio = cv2.contourArea(contour) / max(1.0, area)
            confidence = min(0.96, self.confidence_floor + 0.25 * fill_ratio + min(0.3, bbox_area(bbox.xyxy) / 8000.0))
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=float(confidence),
                    class_name="player",
                    attributes={"backend": "opencv_color_motion"},
                )
            )
        detections.sort(key=lambda det: (det.bbox.y1, det.bbox.x1))
        return detections


class UltralyticsPlayerDetector(PlayerDetector):
    """Optional YOLO adapter. It is loaded only when ultralytics is installed."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        person_class_id: int = 0,
        device: str | None = None,
        image_size: int | None = None,
        use_tracking: bool = False,
        tracker_config: str = "botsort.yaml",
        nms_iou_threshold: float = 0.55,
        containment_threshold: float = 0.78,
        min_bbox_height: float = 0.0,
        min_bbox_width: float = 0.0,
        require_playing_surface: bool = False,
        min_playing_surface_fraction: float = 0.18,
        reject_bottom_edge_detections: bool = False,
        bottom_edge_margin_pixels: float = 2.0,
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("ultralytics is not installed") from exc
        self.model = YOLO(weights)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.device = device
        self.image_size = image_size
        self.use_tracking = use_tracking
        self.tracker_config = tracker_config
        self.nms_iou_threshold = nms_iou_threshold
        self.containment_threshold = containment_threshold
        self.min_bbox_height = min_bbox_height
        self.min_bbox_width = min_bbox_width
        self.require_playing_surface = require_playing_surface
        self.min_playing_surface_fraction = min_playing_surface_fraction
        self.reject_bottom_edge_detections = reject_bottom_edge_detections
        self.bottom_edge_margin_pixels = bottom_edge_margin_pixels

    def detect(self, frame: np.ndarray, frame_index: int, timestamp: float) -> list[Detection]:
        predict_kwargs = {
            "conf": self.confidence_threshold,
            "verbose": False,
            "device": self.device,
            "classes": [self.person_class_id],
        }
        if self.image_size:
            predict_kwargs["imgsz"] = self.image_size
        if self.use_tracking:
            results = self.model.track(frame, persist=True, tracker=self.tracker_config, **predict_kwargs)
        else:
            results = self.model.predict(frame, **predict_kwargs)
        detections: list[Detection] = []
        height, width = frame.shape[:2]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            kp_xy = None
            kp_conf = None
            keypoints_obj = getattr(result, "keypoints", None)
            if keypoints_obj is not None:
                xy = getattr(keypoints_obj, "xy", None)
                cf = getattr(keypoints_obj, "conf", None)
                if xy is not None:
                    kp_xy = xy.detach().cpu().numpy()
                if cf is not None:
                    kp_conf = cf.detach().cpu().numpy()
            # Optional segmentation masks (YOLO11m-seg). Used downstream to wipe grass
            # from torso crops so PARSeq sees only the player.
            masks_np = None
            masks_obj = getattr(result, "masks", None)
            if masks_obj is not None and getattr(masks_obj, "data", None) is not None:
                masks_np = masks_obj.data.detach().cpu().numpy()  # (N, mH, mW) uint8
            for idx, box in enumerate(boxes):
                cls = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                if cls != self.person_class_id:
                    continue
                conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                bbox = BBox(*map(float, xyxy)).clipped(width, height)
                if bbox.height < self.min_bbox_height or bbox.width < self.min_bbox_width:
                    continue
                if self.reject_bottom_edge_detections and bbox.y2 >= height - self.bottom_edge_margin_pixels:
                    continue
                if self.require_playing_surface and not _bbox_has_playing_surface(
                    frame, bbox, min_fraction=self.min_playing_surface_fraction
                ):
                    continue
                track_id = None
                if getattr(box, "id", None) is not None:
                    try:
                        track_id = int(box.id.item()) if hasattr(box.id, "item") else int(box.id)
                    except Exception:
                        track_id = None
                attrs = {"backend": "ultralytics_yolo", "track_id": track_id}
                if kp_xy is not None and idx < kp_xy.shape[0]:
                    keypoints = kp_xy[idx].tolist()
                    confs = kp_conf[idx].tolist() if kp_conf is not None and idx < kp_conf.shape[0] else [0.0] * len(keypoints)
                    attrs["pose_keypoints"] = keypoints
                    attrs["pose_keypoint_conf"] = confs
                if masks_np is not None and idx < masks_np.shape[0]:
                    # Upscale mask to frame size (the seg head outputs at network resolution).
                    mask_small = masks_np[idx]
                    if mask_small.shape != (height, width):
                        mask_full = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_full = mask_small
                    attrs["segmentation_mask"] = (mask_full > 0).astype(np.uint8)
                detections.append(
                    Detection(
                        bbox=bbox,
                        confidence=conf,
                        class_name="player",
                        attributes=attrs,
                    )
                )
        return _suppress_duplicate_detections(
            detections,
            iou_threshold=self.nms_iou_threshold,
            containment_threshold=self.containment_threshold,
        )


def _suppress_duplicate_detections(
    detections: list[Detection],
    iou_threshold: float = 0.55,
    containment_threshold: float = 0.78,
) -> list[Detection]:
    """Remove near-duplicate person boxes before tracking.

    Broadcast lacrosse clusters often produce one clean full-body box plus smaller
    partial boxes on the same player. If those all reach the tracker, each becomes an
    active track and the renderer draws overlapping labels for one athlete.
    """
    if len(detections) <= 1:
        return detections
    ordered = sorted(detections, key=lambda det: det.confidence, reverse=True)
    kept: list[Detection] = []
    for det in ordered:
        duplicate = False
        det_area = max(1.0, det.bbox.area)
        for prev in kept:
            if iou_xyxy(det.bbox.xyxy, prev.bbox.xyxy) >= iou_threshold:
                duplicate = True
                break
            ix1 = max(det.bbox.x1, prev.bbox.x1)
            iy1 = max(det.bbox.y1, prev.bbox.y1)
            ix2 = min(det.bbox.x2, prev.bbox.x2)
            iy2 = min(det.bbox.y2, prev.bbox.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / det_area >= containment_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    kept.sort(key=lambda det: (det.bbox.y1, det.bbox.x1))
    return kept


def _bbox_has_playing_surface(
    frame: np.ndarray,
    bbox: BBox,
    min_fraction: float = 0.18,
) -> bool:
    """Return True when a detection's base sits on the lacrosse floor.

    This filters spectators: a COCO person detector cannot tell players from crowd,
    but audience boxes usually do not have green turf, purple crease paint, or white
    floor markings directly under/around their feet.
    """
    h, w = frame.shape[:2]
    bw = max(1.0, bbox.width)
    bh = max(1.0, bbox.height)
    x1 = max(0, int(round(bbox.x1 - 0.25 * bw)))
    x2 = min(w, int(round(bbox.x2 + 0.25 * bw)))
    y1 = max(0, int(round(bbox.y2 - 0.10 * bh)))
    y2 = min(h, int(round(bbox.y2 + 0.18 * bh)))
    if x2 <= x1 or y2 <= y1:
        return False
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green_floor = (hue >= 34) & (hue <= 96) & (sat >= 32) & (val >= 35)
    purple_floor = (hue >= 124) & (hue <= 166) & (sat >= 25) & (val >= 45)
    white_marking = (sat <= 70) & (val >= 150)
    floor_like = green_floor | purple_floor | white_marking
    return float(np.mean(floor_like)) >= min_fraction


def build_player_detector(config: dict[str, Any]) -> PlayerDetector:
    detector_config = config.get("detector", {})
    backend = str(detector_config.get("backend", "auto")).lower()
    if backend in {"auto", "ultralytics", "yolo"}:
        weights = detector_config.get("weights")
        if weights or backend in {"ultralytics", "yolo"}:
            try:
                return UltralyticsPlayerDetector(
                    weights=weights or "yolo11n.pt",
                    confidence_threshold=float(detector_config.get("confidence_threshold", 0.25)),
                    device=detector_config.get("device"),
                    image_size=detector_config.get("image_size"),
                    use_tracking=bool(detector_config.get("use_tracking", False)),
                    tracker_config=str(detector_config.get("tracker_config", "botsort.yaml")),
                    nms_iou_threshold=float(detector_config.get("nms_iou_threshold", 0.55)),
                    containment_threshold=float(detector_config.get("containment_threshold", 0.78)),
                    min_bbox_height=float(detector_config.get("min_bbox_height", 0.0)),
                    min_bbox_width=float(detector_config.get("min_bbox_width", 0.0)),
                    require_playing_surface=bool(detector_config.get("require_playing_surface", False)),
                    min_playing_surface_fraction=float(detector_config.get("min_playing_surface_fraction", 0.18)),
                    reject_bottom_edge_detections=bool(detector_config.get("reject_bottom_edge_detections", False)),
                    bottom_edge_margin_pixels=float(detector_config.get("bottom_edge_margin_pixels", 2.0)),
                )
            except Exception:
                if backend != "auto":
                    raise

    opencv_config = detector_config.get("opencv", {})
    return OpenCVPlayerDetector(
        min_area=int(opencv_config.get("player_min_area", 300)),
        max_area_ratio=float(opencv_config.get("player_max_area_ratio", 0.08)),
        min_height=int(opencv_config.get("player_min_height", 22)),
        min_aspect=float(opencv_config.get("player_min_aspect", 0.22)),
        max_aspect=float(opencv_config.get("player_max_aspect", 1.25)),
        confidence_floor=float(opencv_config.get("confidence_floor", 0.35)),
    )
