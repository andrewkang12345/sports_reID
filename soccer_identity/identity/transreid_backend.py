from __future__ import annotations

import collections.abc
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


class TransReIDBoxmotBackend:
    """Expose an official TransReID checkpoint through BoxMOT's ReID interface."""

    def __init__(
        self,
        repo: str,
        weights: str,
        config: str,
        device: str = "cuda:0",
        batch_size: int = 32,
    ) -> None:
        self.repo = Path(repo).expanduser().resolve()
        self.weights = Path(weights).expanduser().resolve()
        self.config = Path(config).expanduser().resolve()
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.feature_dimension = 3840

        for path in (self.repo, self.weights, self.config):
            if not path.exists():
                raise FileNotFoundError(f"TransReID dependency not found: {path}")

        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        if "torch._six" not in sys.modules:
            torch_six = types.ModuleType("torch._six")
            torch_six.container_abcs = collections.abc
            sys.modules["torch._six"] = torch_six

        from config import cfg
        from model import make_model

        cfg.merge_from_file(str(self.config))
        cfg.defrost()
        cfg.MODEL.PRETRAIN_CHOICE = "none"
        cfg.MODEL.PRETRAIN_PATH = ""
        cfg.freeze()

        model = make_model(cfg, num_class=1041, camera_num=15, view_num=1)
        model.load_param(str(self.weights))
        return model.to(self.device).eval()

    def get_features(self, xyxys: np.ndarray, image: np.ndarray) -> np.ndarray:
        if len(xyxys) == 0:
            return np.empty((0, self.feature_dimension), dtype=np.float32)

        height, width = image.shape[:2]
        crops: list[np.ndarray] = []
        for x1, y1, x2, y2 in xyxys:
            left = max(0, min(width - 1, int(np.floor(x1))))
            top = max(0, min(height - 1, int(np.floor(y1))))
            right = max(left + 1, min(width, int(np.ceil(x2))))
            bottom = max(top + 1, min(height, int(np.ceil(y2))))
            crops.append(image[top:bottom, left:right])

        output: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                tensor = self._prepare_batch(crops[start : start + self.batch_size])
                camera = torch.zeros(
                    len(tensor), dtype=torch.long, device=self.device
                )
                features = self.model(tensor, cam_label=camera)
                features = torch.nn.functional.normalize(features.float(), dim=1)
                output.append(features.cpu().numpy())
        return np.concatenate(output).astype(np.float32, copy=False)

    def _prepare_batch(self, crops: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_CUBIC)
            normalized = resized.astype(np.float32) / 127.5 - 1.0
            tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1)))
        return torch.stack(tensors).to(self.device)
