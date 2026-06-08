"""Wrapper around boxmot's ReID backend for extracting per-detection appearance embeddings.

The body_reid module's existing `image_embedding` is a 32x64 HSV histogram — useful as a
weak fallback but inadequate for cross-track ReID matching across occlusions and camera
cuts. This module loads a real ReID network (default: OSNet x1.0 trained on MSMT17) and
returns 512-D embeddings that can be cosine-compared across tracks.

Used after tracklet construction for appearance-memory-based identity override: for each
track, snapshot embeddings on its top-K OCR-confident frames, cluster across tracks, and
re-assign jerseys for tracks whose appearance matches a high-confidence anchor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


_EXTRACTOR_CACHE: dict[str, Any] = {}


def load_reid_extractor(
    weights: str = "osnet_x1_0_msmt17.pt",
    device: str = "cuda:0",
    half: bool = False,
):
    """Lazy-load and cache a single ReID extractor instance."""
    key = f"{weights}|{device}|{half}"
    if key in _EXTRACTOR_CACHE:
        return _EXTRACTOR_CACHE[key]
    try:
        import torch
        from boxmot.appearance.reid_auto_backend import ReidAutoBackend
    except Exception as exc:  # pragma: no cover - optional dep
        raise ImportError("boxmot is required for ReID extraction") from exc

    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = Path.cwd() / weights_path
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    backend = ReidAutoBackend(weights=weights_path, device=dev, half=half)
    backend.model.warmup()
    _EXTRACTOR_CACHE[key] = backend.model
    return backend.model


def extract_appearance_batch(
    extractor,
    frame: np.ndarray,
    bboxes: list[tuple[float, float, float, float]],
) -> list[np.ndarray] | None:
    """Run ReID on all bboxes in one shot. Returns one (D,) ndarray per box.

    `frame` is BGR (or RGB — boxmot's preprocessing handles either since the model was
    trained on cropped pedestrians; we keep BGR consistent with cv2.imread).
    """
    if not bboxes:
        return []
    try:
        boxes_arr = np.asarray(bboxes, dtype=np.float32)
        feats = extractor.get_features(boxes_arr, frame)
        if feats is None:
            return None
        # feats: (N, D). Normalize to unit length for cosine similarity later.
        feats = np.asarray(feats, dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        feats = feats / norms
        return [feats[i] for i in range(feats.shape[0])]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[reid] extraction failed: {exc}")
        return None
