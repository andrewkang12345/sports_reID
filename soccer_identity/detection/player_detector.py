from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from soccer_identity.utils.geometry import bbox_area
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
        return detections


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
