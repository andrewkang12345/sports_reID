"""MixSort tracker wrapper.

Wraps the official MixSort (MCG-NJU/MixSort, CVPR 2023, SportsMOT baseline) tracker so
it implements the same MultiObjectTracker interface as our existing BoxmotBoTSORTTracker.

MixSort augments a Kalman+IoU base (ByteTrack or OC-SORT style) with a MixFormer-DeiT
template-matching head for appearance association. On SportsMOT it edges out ByteTrack by
~1.6 HOTA and OC-SORT by ~0.4. Whether that translates to better identity coverage on a
30-second broadcast clip is what we're measuring here.

Setup:
  third_party/MixViT/      — the MixViT package (Python source)
  third_party/mixsort/     — the mixsort_tracker / mixsort_oc_tracker modules
  models/mixsort/MixFormer_soccernet_train.pth.tar — trained MixFormer weights
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from soccer_identity.utils.schemas import BBox, Detection
from soccer_identity.tracking.tracker import MultiObjectTracker, TrackedDetection


def _ensure_mixsort_paths(root: Path) -> None:
    """Ensure the third_party paths are importable. Called lazily on tracker init."""
    mixvit = root / "third_party" / "MixViT"
    mixsort = root / "third_party"
    for p in (str(mixvit), str(mixsort)):
        if p not in sys.path:
            sys.path.insert(0, p)


class MixSortTracker(MultiObjectTracker):
    """MixSort-with-MixFormer tracker, soccernet-trained appearance head.

    The upstream MixSort tracker expects a YOLOX-shaped detection tensor. We adapt our
    Detection objects to that shape and stamp the original Detection back onto each
    returned TrackedDetection so downstream code (kit color, pose, OCR) still sees the
    same detection it would with any other tracker backend.
    """

    def __init__(
        self,
        mixformer_weights: str,
        device: str = "cuda:0",
        track_thresh: float = 0.5,
        track_buffer: int = 60,
        match_thresh: float = 0.8,
        alpha: float = 0.6,
        radius: int = 0,
        iou_thresh: float = 0.3,
        script: str = "mixformer_deit",
        config: str = "soccernet",
        frame_rate: int = 30,
        mot20: bool = False,
        project_root: str | None = None,
    ) -> None:
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        _ensure_mixsort_paths(root)

        # Build args namespace matching the upstream tracker's expectations.
        args = SimpleNamespace(
            track_thresh=float(track_thresh),
            track_buffer=int(track_buffer),
            match_thresh=float(match_thresh),
            alpha=float(alpha),
            radius=int(radius),
            iou_thresh=float(iou_thresh),
            script=str(script),
            config=str(config),
            mot20=bool(mot20),
            local_rank=int(device.split(":")[-1]) if device.startswith("cuda:") else 0,
        )
        # Tell the upstream code where the config lives. It hard-codes a relative path
        # ../../MixViT — we patch that by injecting cfg_file directly into Settings.
        self._cfg_file = str(root / "third_party" / "MixViT" / "experiments" / script / f"{config}.yaml")

        # Import MixSort tracker AFTER paths are set
        from mixsort.mixsort_tracker.mixsort_tracker import MIXTracker as MixTracker

        # The upstream MixTracker.__init__ recomputes prj_dir relative to its own file
        # which would land in third_party/mixsort/mixsort_tracker — that's correct since
        # we copied the entire MixViT alongside it. So no further patching needed for
        # paths. But it expects pretrained weights under MODEL.BACKBONE.PRETRAINED_PATH;
        # we load them manually after construction to use our SoccerNet-trained head.
        self.tracker = MixTracker(args, frame_rate=frame_rate)

        # Load the trained MixFormer state_dict over the freshly initialized backbone
        import torch
        ckpt = torch.load(mixformer_weights, map_location=device, weights_only=False)
        sd = ckpt.get("net", ckpt.get("model", ckpt.get("state_dict", ckpt)))
        missing, unexpected = self.tracker.network.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[MixSort] loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print(f"[MixSort] loaded trained weights (all keys matched)")
        self.tracker.network.eval()
        self._device = device

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        if frame is None or not detections:
            return []
        import torch as _torch
        import cv2 as _cv2

        # MixSort expects (N, 5) ndarray [x1, y1, x2, y2, score] in original image coords.
        rows = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox.xyxy
            rows.append([x1, y1, x2, y2, float(det.confidence)])
        dets = np.asarray(rows, dtype=np.float32)

        img_h, img_w = frame.shape[:2]
        # MixSort's crop_and_resize calls torchvision.resized_crop on `img` and then
        # normalizes against ImageNet RGB stats. Pass a CHW float tensor in RGB on GPU.
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        img_tensor = _torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(self._device)
        out_tracks = self.tracker.update(
            dets,
            (img_h, img_w),
            (img_h, img_w),
            img_tensor,
        )

        # Map each returned STrack back to the source Detection by max IoU.
        tracked: list[TrackedDetection] = []
        used = set()
        for strack in out_tracks:
            try:
                bb = strack.tlbr  # [x1, y1, x2, y2]
                tid = int(strack.track_id)
                score = float(getattr(strack, "score", 1.0))
            except Exception:
                continue
            # Find best-matching source detection (max IoU)
            best_di = -1
            best_iou = 0.0
            for di, det in enumerate(detections):
                if di in used:
                    continue
                dx1, dy1, dx2, dy2 = det.bbox.xyxy
                inter_x1 = max(bb[0], dx1)
                inter_y1 = max(bb[1], dy1)
                inter_x2 = min(bb[2], dx2)
                inter_y2 = min(bb[3], dy2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter = inter_w * inter_h
                a1 = (bb[2] - bb[0]) * (bb[3] - bb[1])
                a2 = (dx2 - dx1) * (dy2 - dy1)
                u = a1 + a2 - inter
                iou = inter / u if u > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_di = di
            if best_di < 0 or best_iou < 0.20:
                continue
            used.add(best_di)
            detection = detections[best_di]
            bbox = BBox(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])).clipped(int(img_w), int(img_h))
            tracked.append(TrackedDetection(str(tid), bbox, score, detection))
        return tracked
