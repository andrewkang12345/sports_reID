from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from soccer_identity.utils.geometry import crop_xyxy
from soccer_identity.utils.schemas import Detection


class PlayerSegmenter:
    """Interface for player segmentation.

    Production deployments can replace this with SAM/SAM2, Sapiens, Detectron2,
    or another model. The fallback returns a foreground-ish mask inside the box.
    """

    def segment(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        raise NotImplementedError


class BoxMaskSegmenter(PlayerSegmenter):
    def segment(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        for detection in detections:
            crop = crop_xyxy(frame, detection.bbox.xyxy)
            if crop.size == 0:
                masks.append(np.zeros(frame.shape[:2], dtype=np.uint8))
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            green = (hue >= 35) & (hue <= 90) & (sat >= 35) & (val >= 35)
            local_mask = (~green & (val > 20)).astype(np.uint8) * 255
            local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            full = np.zeros(frame.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = map(int, detection.bbox.xyxy)
            full[y1:y2, x1:x2] = local_mask[: max(0, y2 - y1), : max(0, x2 - x1)]
            masks.append(full)
        return masks


def build_player_segmenter(config: dict[str, Any]) -> PlayerSegmenter:
    return BoxMaskSegmenter()
