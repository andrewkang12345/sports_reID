from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_log(value: float, eps: float = 1e-6) -> float:
    return math.log(max(float(value), eps))


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32) / max(float(temperature), 1e-6)
    if arr.size == 0:
        return arr
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    total = np.sum(exp)
    if total <= 0:
        return np.ones_like(exp) / max(1, exp.size)
    return exp / total


def cosine_similarity(a: np.ndarray | Sequence[float], b: np.ndarray | Sequence[float]) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    if av.size == 0 or bv.size == 0 or av.size != bv.size:
        return 0.0
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(av, bv) / denom)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def crop_xyxy(frame: np.ndarray, xyxy: Sequence[float], pad: int = 0) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def bbox_center(xyxy: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_bottom_center(xyxy: Sequence[float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = map(float, xyxy)
    return ((x1 + x2) * 0.5, y2)


def bbox_area(xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, xyxy)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def merge_probability_dicts(dicts: Iterable[dict[str, float]]) -> dict[str, float]:
    accum: dict[str, float] = {}
    count = 0
    for item in dicts:
        if not item:
            continue
        count += 1
        for key, value in item.items():
            accum[key] = accum.get(key, 0.0) + float(value)
    if count == 0:
        return {}
    total = sum(accum.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in accum.items()}


def dominant_non_green_rgb(crop: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Return mean RGB of saturated non-field pixels and a simple quality score."""
    if crop.size == 0:
        return None, 0.0
    h, w = crop.shape[:2]
    if h < 8 or w < 4:
        return None, 0.0
    upper = crop[int(h * 0.15) : int(h * 0.72), int(w * 0.12) : int(w * 0.88)]
    if upper.size == 0:
        upper = crop
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green = (hue >= 35) & (hue <= 90) & (sat >= 45) & (val >= 40)
    saturated = (sat >= 35) & (val >= 35)
    mask = saturated & ~green
    if np.count_nonzero(mask) < 8:
        mask = val > 30
    if np.count_nonzero(mask) == 0:
        return None, 0.0
    bgr = upper[mask].reshape(-1, 3).astype(np.float32)
    rgb = bgr[:, ::-1].mean(axis=0)
    quality = float(clamp(np.count_nonzero(mask) / max(1, upper.shape[0] * upper.shape[1]), 0.0, 1.0))
    return rgb.astype(np.float32), quality


def dominant_non_green_rgb_region(
    crop: np.ndarray,
    y_start: float,
    y_end: float,
    x_start: float = 0.12,
    x_end: float = 0.88,
) -> tuple[np.ndarray | None, float]:
    if crop.size == 0:
        return None, 0.0
    h, w = crop.shape[:2]
    y1 = int(h * y_start)
    y2 = int(h * y_end)
    x1 = int(w * x_start)
    x2 = int(w * x_end)
    region = crop[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
    if region.size == 0:
        return None, 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green = (hue >= 35) & (hue <= 90) & (sat >= 45) & (val >= 40)
    visible = val >= 28
    saturated_or_uniform = ((sat >= 28) | (val >= 115) | (val <= 80)) & visible
    mask = saturated_or_uniform & ~green
    if np.count_nonzero(mask) < 8:
        return None, 0.0
    rgb = region[mask].reshape(-1, 3).astype(np.float32)[:, ::-1].mean(axis=0)
    quality = float(clamp(np.count_nonzero(mask) / max(1, region.shape[0] * region.shape[1]), 0.0, 1.0))
    return rgb.astype(np.float32), quality


def parse_hex_color(value: str) -> np.ndarray:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {value!r}")
    return np.asarray([int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)], dtype=np.float32)


def image_embedding(crop: np.ndarray, size: tuple[int, int] = (32, 64)) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((size[0] * size[1] * 3,), dtype=np.float32)
    resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    hist = hist.astype(np.float32)
    return l2_normalize(hist)


def crop_quality(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    area_score = clamp((h * w) / (80.0 * 160.0), 0.0, 1.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = clamp(sharpness / 150.0, 0.0, 1.0)
    return float(0.55 * area_score + 0.45 * sharpness_score)
