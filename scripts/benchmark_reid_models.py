#!/usr/bin/env python3
"""Benchmark appearance and gait ReID models on the ARG-FRA tracking ground truth.

The frame-level protocol uses the exact annotated player boxes. It measures:

* leave-one-frame-out retrieval over all player crops;
* retrieval of "for tracker" crops against number-visible crops;
* one-to-one appearance assignment between adjacent annotated frames; and
* temporal retrieval after splitting each player's observations into two sequences.

OpenGait is evaluated only with the temporal protocol because it consumes a sequence
of silhouettes rather than independent RGB crops.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = ROOT / "groundTruth_AllTracking_ARG_FRA_183303"
DEFAULT_EXTERNAL = Path("/mnt/data/reid_models")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ascii_normalize(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def parse_category(name: str) -> tuple[str | None, bool, bool]:
    name = name.strip()
    if name in {"Player-Detection", "referee"}:
        return None, False, True
    if name.startswith("for tracker "):
        match = re.match(r"for tracker \d+\s+(.+)", name)
        return (match.group(1).strip() if match else None), False, False
    if name.startswith("Goalkeeper "):
        match = re.match(r"Goalkeeper \d+\s+(.+)", name)
        return (match.group(1).strip() if match else None), True, False
    match = re.match(r"\d+\s+(.+)", name)
    return (match.group(1).strip() if match else None), True, False


def frame_number(file_name: str) -> int:
    match = re.search(r"mp4-(\d{4})", file_name)
    return int(match.group(1)) if match else -1


def load_dataset(gt_dir: Path) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    coco = json.loads((gt_dir / "_annotations.coco.json").read_text())
    images = {item["id"]: item for item in coco["images"]}
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    frames: dict[int, np.ndarray] = {}
    samples: list[dict[str, Any]] = []

    for ann in coco["annotations"]:
        identity, visible, role = parse_category(categories[ann["category_id"]])
        if role or identity is None:
            continue
        image = images[ann["image_id"]]
        path = gt_dir / image["file_name"]
        if ann["image_id"] not in frames:
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"Could not read ground-truth frame: {path}")
            frames[ann["image_id"]] = frame
        frame = frames[ann["image_id"]]
        x, y, width, height = ann["bbox"]
        x1 = max(0, int(math.floor(x)))
        y1 = max(0, int(math.floor(y)))
        x2 = min(frame.shape[1], int(math.ceil(x + width)))
        y2 = min(frame.shape[0], int(math.ceil(y + height)))
        if x2 <= x1 or y2 <= y1:
            continue
        samples.append(
            {
                "annotation_id": ann["id"],
                "image_id": ann["image_id"],
                "frame_number": frame_number(image["file_name"]),
                "identity": ascii_normalize(identity),
                "visible": bool(visible),
                "bbox": [x1, y1, x2, y2],
                "crop": frame[y1:y2, x1:x2].copy(),
            }
        )

    samples.sort(
        key=lambda item: (
            item["frame_number"],
            item["identity"],
            item["annotation_id"],
        )
    )
    return samples, frames


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def average_precision(ranked_matches: np.ndarray) -> float:
    positives = int(ranked_matches.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(ranked_matches) / np.arange(1, len(ranked_matches) + 1)
    return float((precision * ranked_matches).sum() / positives)


def retrieval_metrics(
    similarities: np.ndarray,
    query_indices: list[int],
    gallery_indices_for_query,
    identities: list[str],
) -> dict[str, Any]:
    rank1 = 0
    rank5 = 0
    average_precisions: list[float] = []
    eligible = 0
    for query_index in query_indices:
        gallery_indices = list(gallery_indices_for_query(query_index))
        positive_count = sum(
            identities[index] == identities[query_index] for index in gallery_indices
        )
        if not gallery_indices or positive_count == 0:
            continue
        scores = similarities[query_index, gallery_indices]
        order = np.argsort(-scores)
        ranked_matches = np.asarray(
            [
                identities[gallery_indices[index]] == identities[query_index]
                for index in order
            ],
            dtype=np.float32,
        )
        eligible += 1
        rank1 += int(ranked_matches[0] == 1)
        rank5 += int(ranked_matches[:5].max(initial=0) == 1)
        average_precisions.append(average_precision(ranked_matches))
    return {
        "queries": eligible,
        "rank1_pct": 100.0 * rank1 / max(1, eligible),
        "rank5_pct": 100.0 * rank5 / max(1, eligible),
        "map_pct": 100.0 * float(np.nanmean(average_precisions))
        if average_precisions
        else 0.0,
    }


def temporal_centroids(
    features: np.ndarray, samples: list[dict[str, Any]], minimum_observations: int = 6
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    by_identity: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_identity[sample["identity"]].append(index)

    gallery: list[np.ndarray] = []
    query: list[np.ndarray] = []
    labels: list[str] = []
    for identity, indices in sorted(by_identity.items()):
        indices.sort(key=lambda index: samples[index]["frame_number"])
        if len(indices) < minimum_observations:
            continue
        gallery_indices = indices[::2]
        query_indices = indices[1::2]
        if not gallery_indices or not query_indices:
            continue
        gallery.append(normalize_rows(features[gallery_indices].mean(axis=0)[None])[0])
        query.append(normalize_rows(features[query_indices].mean(axis=0)[None])[0])
        labels.append(identity)
    return np.asarray(query), np.asarray(gallery), labels


def temporal_retrieval(
    features: np.ndarray, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    query, gallery, labels = temporal_centroids(features, samples)
    if len(labels) == 0:
        return {"identities": 0, "rank1_pct": 0.0, "map_pct": 0.0}
    similarities = query @ gallery.T
    rank1 = 0
    average_precisions = []
    for index, identity in enumerate(labels):
        order = np.argsort(-similarities[index])
        matches = np.asarray([labels[item] == identity for item in order], dtype=float)
        rank1 += int(matches[0] == 1)
        average_precisions.append(average_precision(matches))
    return {
        "identities": len(labels),
        "rank1_pct": 100.0 * rank1 / len(labels),
        "map_pct": 100.0 * float(np.mean(average_precisions)),
    }


def adjacent_assignment(
    similarities: np.ndarray, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_frame[sample["frame_number"]].append(index)
    frames = sorted(by_frame)
    correct = 0
    assignments = 0
    for left_frame, right_frame in zip(frames, frames[1:]):
        left = by_frame[left_frame]
        right = by_frame[right_frame]
        rows, columns = linear_sum_assignment(-similarities[np.ix_(left, right)])
        for row, column in zip(rows, columns):
            assignments += 1
            correct += int(
                samples[left[row]]["identity"] == samples[right[column]]["identity"]
            )
    return {
        "assignments": assignments,
        "correct": correct,
        "accuracy_pct": 100.0 * correct / max(1, assignments),
    }


def evaluate_frame_features(
    features: np.ndarray, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    features = normalize_rows(features)
    similarities = features @ features.T
    identities = [sample["identity"] for sample in samples]
    frame_numbers = [sample["frame_number"] for sample in samples]
    visible = [sample["visible"] for sample in samples]
    all_indices = list(range(len(samples)))

    all_retrieval = retrieval_metrics(
        similarities,
        all_indices,
        lambda query: (
            index
            for index in all_indices
            if frame_numbers[index] != frame_numbers[query]
        ),
        identities,
    )
    tracker_indices = [index for index, is_visible in enumerate(visible) if not is_visible]
    tracker_to_visible = retrieval_metrics(
        similarities,
        tracker_indices,
        lambda query: (
            index
            for index in all_indices
            if visible[index] and frame_numbers[index] != frame_numbers[query]
        ),
        identities,
    )
    return {
        "all_crop_retrieval": all_retrieval,
        "tracker_to_visible_retrieval": tracker_to_visible,
        "adjacent_frame_assignment": adjacent_assignment(similarities, samples),
        "temporal_split_retrieval": temporal_retrieval(features, samples),
    }


def extract_hsv(samples: list[dict[str, Any]]) -> np.ndarray:
    from soccer_identity.utils.geometry import image_embedding

    return np.asarray([image_embedding(sample["crop"]) for sample in samples])


def extract_boxmot(
    samples: list[dict[str, Any]],
    frames: dict[int, np.ndarray],
    weights: Path,
    device: str,
) -> np.ndarray:
    from soccer_identity.identity.reid_extractor import (
        extract_appearance_batch,
        load_reid_extractor,
    )

    extractor = load_reid_extractor(str(weights), device=device, half=False)
    by_image: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_image[sample["image_id"]].append(index)
    output: list[np.ndarray | None] = [None] * len(samples)
    for image_id, indices in by_image.items():
        boxes = [samples[index]["bbox"] for index in indices]
        features = extract_appearance_batch(extractor, frames[image_id], boxes)
        if features is None or len(features) != len(indices):
            raise RuntimeError(f"BoxMOT extraction failed for image {image_id}")
        for index, feature in zip(indices, features):
            output[index] = feature
    if any(feature is None for feature in output):
        raise RuntimeError("BoxMOT did not return every requested feature")
    return np.asarray(output, dtype=np.float32)


def batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_dinov2(
    samples: list[dict[str, Any]], device: str, batch_size: int
) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    model_name = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    output = []
    with torch.inference_mode():
        for batch in batches(samples, batch_size):
            images = [
                Image.fromarray(cv2.cvtColor(sample["crop"], cv2.COLOR_BGR2RGB))
                for sample in batch
            ]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = model(**inputs).last_hidden_state[:, 0]
            output.append(features.float().cpu().numpy())
    return np.concatenate(output)


def prepare_tensor_batch(
    samples: list[dict[str, Any]],
    size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    import torch

    tensors = []
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    height, width = size
    for sample in samples:
        rgb = cv2.cvtColor(sample["crop"], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
        normalized = (resized.astype(np.float32) / 255.0 - mean_array) / std_array
        tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1)))
    return torch.stack(tensors)


def extract_transreid(
    samples: list[dict[str, Any]],
    device: str,
    batch_size: int,
    external_root: Path,
) -> np.ndarray:
    import collections.abc
    import torch
    import types

    repo = external_root / "TransReID"
    sys.path.insert(0, str(repo))
    if "torch._six" not in sys.modules:
        torch_six = types.ModuleType("torch._six")
        torch_six.container_abcs = collections.abc
        sys.modules["torch._six"] = torch_six
    if "mmcv.runner" not in sys.modules:
        mmcv = types.ModuleType("mmcv")
        mmcv_runner = types.ModuleType("mmcv.runner")

        def unavailable_mmcv_loader(*_args, **_kwargs):
            raise RuntimeError("MMCV checkpoint loading is not used by this adapter")

        mmcv_runner.load_checkpoint = unavailable_mmcv_loader
        mmcv.runner = mmcv_runner
        sys.modules["mmcv"] = mmcv
        sys.modules["mmcv.runner"] = mmcv_runner
    from config import cfg
    from model import make_model

    cfg.merge_from_file(str(repo / "configs/MSMT17/vit_transreid_stride.yml"))
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.MODEL.PRETRAIN_PATH = ""
    cfg.freeze()
    model = make_model(cfg, num_class=1041, camera_num=15, view_num=1)
    weights = external_root / "checkpoints/transreid_msmt17_vit.pth"
    model.load_param(str(weights))
    model = model.to(device).eval()
    output = []
    with torch.inference_mode():
        for batch in batches(samples, batch_size):
            tensor = prepare_tensor_batch(
                batch, (256, 128), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ).to(device)
            camera = torch.zeros(len(batch), dtype=torch.long, device=device)
            features = model(tensor, cam_label=camera)
            output.append(features.float().cpu().numpy())
    return np.concatenate(output)


def extract_solider(
    samples: list[dict[str, Any]],
    device: str,
    batch_size: int,
    external_root: Path,
) -> np.ndarray:
    import collections.abc
    import torch
    import types

    repo = external_root / "SOLIDER-REID"
    sys.path.insert(0, str(repo))
    if "torch._six" not in sys.modules:
        torch_six = types.ModuleType("torch._six")
        torch_six.container_abcs = collections.abc
        sys.modules["torch._six"] = torch_six
    if "mmcv.runner" not in sys.modules:
        mmcv = types.ModuleType("mmcv")
        mmcv_runner = types.ModuleType("mmcv.runner")

        def unavailable_mmcv_loader(*_args, **_kwargs):
            raise RuntimeError("MMCV checkpoint loading is not used by this adapter")

        mmcv_runner.load_checkpoint = unavailable_mmcv_loader
        mmcv.runner = mmcv_runner
        sys.modules["mmcv"] = mmcv
        sys.modules["mmcv.runner"] = mmcv_runner
    from config import cfg
    from model import make_model

    cfg.merge_from_file(str(repo / "configs/msmt17/swin_small.yml"))
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.MODEL.PRETRAIN_PATH = ""
    cfg.freeze()
    model = make_model(
        cfg,
        num_class=1041,
        camera_num=15,
        view_num=1,
        semantic_weight=0.2,
    )
    weights = external_root / "checkpoints/solider_swin_small_msmt17.pth"
    model.load_param(str(weights))
    model = model.to(device).eval()
    output = []
    with torch.inference_mode():
        for batch in batches(samples, batch_size):
            tensor = prepare_tensor_batch(
                batch, (384, 128), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ).to(device)
            features = model(tensor)[0]
            output.append(features.float().cpu().numpy())
    return np.concatenate(output)


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.zeros((len(left), len(right)), dtype=np.float32)
    for row, box_a in enumerate(left):
        for column, box_b in enumerate(right):
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
            area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
            output[row, column] = intersection / max(
                1e-12, area_a + area_b - intersection
            )
    return output


def heuristic_silhouette(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = (
        (hsv[:, :, 0] >= 30)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 25)
    )
    mask = (~green).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    return mask


def align_silhouette(mask: np.ndarray, height: int = 64, width: int = 44) -> np.ndarray:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return np.zeros((height, width), dtype=np.float32)
    x, y, w, h = cv2.boundingRect(points)
    tight = mask[y : y + h, x : x + w]
    resized_width = max(1, int(round(width * tight.shape[1] / max(1, tight.shape[0]))))
    resized = cv2.resize(
        tight, (resized_width, height), interpolation=cv2.INTER_NEAREST
    )
    if resized_width > width:
        start = (resized_width - width) // 2
        resized = resized[:, start : start + width]
    else:
        left = (width - resized_width) // 2
        resized = cv2.copyMakeBorder(
            resized, 0, 0, left, width - resized_width - left, cv2.BORDER_CONSTANT
        )
    return (resized.astype(np.float32) / 255.0).clip(0, 1)


def extract_player_silhouettes(
    samples: list[dict[str, Any]],
    frames: dict[int, np.ndarray],
    device: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "yolo11m-seg.pt"))
    by_image: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_image[sample["image_id"]].append(index)
    silhouettes: list[np.ndarray | None] = [None] * len(samples)
    matched = 0

    for image_id, indices in by_image.items():
        frame = frames[image_id]
        result = model.predict(
            frame,
            imgsz=1280,
            conf=0.05,
            iou=0.6,
            classes=[0],
            retina_masks=True,
            device=device,
            verbose=False,
        )[0]
        predicted_boxes = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None and len(result.boxes)
            else np.empty((0, 4), dtype=np.float32)
        )
        polygons = result.masks.xy if result.masks is not None else []
        gt_boxes = np.asarray([samples[index]["bbox"] for index in indices])
        overlaps = iou_matrix(gt_boxes, predicted_boxes)
        for row, index in enumerate(indices):
            x1, y1, x2, y2 = samples[index]["bbox"]
            mask_crop = None
            if predicted_boxes.size:
                best = int(np.argmax(overlaps[row]))
                if overlaps[row, best] >= 0.10 and best < len(polygons):
                    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    polygon = np.rint(polygons[best]).astype(np.int32)
                    if len(polygon) >= 3:
                        cv2.fillPoly(full_mask, [polygon], 255)
                        mask_crop = full_mask[y1:y2, x1:x2]
                        matched += 1
            if mask_crop is None or np.count_nonzero(mask_crop) < 10:
                mask_crop = heuristic_silhouette(samples[index]["crop"])
            silhouettes[index] = align_silhouette(mask_crop)
    return [item for item in silhouettes if item is not None], {
        "segmentation_matches": matched,
        "samples": len(samples),
        "segmentation_match_pct": 100.0 * matched / max(1, len(samples)),
    }


def extract_opengait(
    samples: list[dict[str, Any]],
    frames: dict[int, np.ndarray],
    device: str,
    external_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    import torch
    import yaml

    repo = external_root / "OpenGait"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "opengait"))
    from opengait.modeling.models.baseline import Baseline

    config_path = repo / "configs/gaitbase/gaitbase_da_gait3d.yaml"
    config = yaml.safe_load(config_path.read_text())
    model = Baseline.__new__(Baseline)
    torch.nn.Module.__init__(model)
    model.build_network(config["model_cfg"])
    model.init_parameters()
    checkpoint_path = (
        external_root
        / "checkpoints/opengait/Gait3D/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()

    silhouettes, segmentation = extract_player_silhouettes(samples, frames, device)
    by_identity: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_identity[sample["identity"]].append(index)

    query: list[np.ndarray] = []
    gallery: list[np.ndarray] = []
    labels: list[str] = []
    with torch.inference_mode():
        for identity, indices in sorted(by_identity.items()):
            indices.sort(key=lambda index: samples[index]["frame_number"])
            if len(indices) < 6:
                continue
            halves = [indices[1::2], indices[::2]]
            embeddings = []
            for half in halves:
                sequence = torch.from_numpy(np.asarray([silhouettes[index] for index in half]))
                sequence = sequence.unsqueeze(0).to(device)
                result = model(([sequence], None, None, None, None))
                embedding = result["inference_feat"]["embeddings"]
                embeddings.append(
                    normalize_rows(embedding.flatten(1).float().cpu().numpy())[0]
                )
            query.append(embeddings[0])
            gallery.append(embeddings[1])
            labels.append(identity)
    return (
        np.asarray(query),
        np.asarray(gallery),
        labels,
        segmentation,
    )


def evaluate_opengait(
    samples: list[dict[str, Any]],
    frames: dict[int, np.ndarray],
    device: str,
    external_root: Path,
) -> dict[str, Any]:
    query, gallery, labels, segmentation = extract_opengait(
        samples, frames, device, external_root
    )
    similarities = query @ gallery.T
    rank1 = 0
    average_precisions = []
    for index, identity in enumerate(labels):
        order = np.argsort(-similarities[index])
        matches = np.asarray([labels[item] == identity for item in order], dtype=float)
        rank1 += int(matches[0] == 1)
        average_precisions.append(average_precision(matches))
    return {
        "temporal_split_retrieval": {
            "identities": len(labels),
            "rank1_pct": 100.0 * rank1 / max(1, len(labels)),
            "map_pct": 100.0 * float(np.mean(average_precisions))
            if average_precisions
            else 0.0,
        },
        "silhouette_extraction": segmentation,
        "frame_level_metrics": None,
    }


def extract_named_frame_model(
    model_name: str,
    samples: list[dict[str, Any]],
    frames: dict[int, np.ndarray],
    device: str,
    batch_size: int,
    external_root: Path,
) -> np.ndarray:
    if model_name == "clip":
        return extract_boxmot(
            samples,
            frames,
            ROOT / "clip_market1501.pt",
            device,
        )
    if model_name == "transreid":
        return extract_transreid(samples, device, batch_size, external_root)
    if model_name == "solider":
        return extract_solider(samples, device, batch_size, external_root)
    if model_name == "dinov2":
        return extract_dinov2(samples, device, batch_size)
    if model_name == "osnet_ain":
        return extract_boxmot(
            samples,
            frames,
            ROOT / "osnet_ain_x1_0_msmt17.pt",
            device,
        )
    raise ValueError(f"Unsupported frame model: {model_name}")


def evaluate_temporal_pair(
    primary: str,
    fusion_weights: list[float],
    samples: list[dict[str, Any]],
    frames: dict[int, np.ndarray],
    device: str,
    batch_size: int,
    external_root: Path,
) -> tuple[dict[str, Any], int]:
    primary_features = normalize_rows(
        extract_named_frame_model(
            primary, samples, frames, device, batch_size, external_root
        )
    )
    appearance_query, appearance_gallery, appearance_labels = temporal_centroids(
        primary_features, samples
    )
    gait_query, gait_gallery, gait_labels, segmentation = extract_opengait(
        samples, frames, device, external_root
    )
    if appearance_labels != gait_labels:
        raise RuntimeError("Appearance and OpenGait temporal identities do not align")

    sweep: dict[str, Any] = {}
    for gait_weight in fusion_weights:
        if not 0.0 <= gait_weight <= 1.0:
            raise ValueError("Fusion weights must be between zero and one")
        query = normalize_rows(
            np.concatenate(
                [
                    math.sqrt(1.0 - gait_weight) * appearance_query,
                    math.sqrt(gait_weight) * gait_query,
                ],
                axis=1,
            )
        )
        gallery = normalize_rows(
            np.concatenate(
                [
                    math.sqrt(1.0 - gait_weight) * appearance_gallery,
                    math.sqrt(gait_weight) * gait_gallery,
                ],
                axis=1,
            )
        )
        similarities = query @ gallery.T
        rank1 = 0
        average_precisions = []
        for index, identity in enumerate(appearance_labels):
            order = np.argsort(-similarities[index])
            matches = np.asarray(
                [appearance_labels[item] == identity for item in order], dtype=float
            )
            rank1 += int(matches[0] == 1)
            average_precisions.append(average_precision(matches))
        sweep[f"{gait_weight:.2f}"] = {
            "identities": len(appearance_labels),
            "rank1_pct": 100.0 * rank1 / max(1, len(appearance_labels)),
            "map_pct": 100.0 * float(np.mean(average_precisions)),
        }
    selected_weight, selected_metrics = max(
        sweep.items(),
        key=lambda item: (item[1]["rank1_pct"], item[1]["map_pct"]),
    )
    return (
        {
            "temporal_split_retrieval": selected_metrics,
            "silhouette_extraction": segmentation,
            "frame_level_metrics": None,
            "selected_opengait_weight": float(selected_weight),
            "selection_metric": "temporal_split_retrieval.rank1_pct_then_map_pct",
            "fusion_sweep": sweep,
        },
        int(appearance_query.shape[1] + gait_query.shape[1]),
    )


def evaluate_fusion(
    primary: str,
    secondary: str,
    fusion_weights: list[float],
    samples: list[dict[str, Any]],
    frames: dict[int, np.ndarray],
    device: str,
    batch_size: int,
    external_root: Path,
) -> tuple[dict[str, Any], int]:
    if secondary == "opengait":
        return evaluate_temporal_pair(
            primary,
            fusion_weights,
            samples,
            frames,
            device,
            batch_size,
            external_root,
        )
    primary_features = normalize_rows(
        extract_named_frame_model(
            primary, samples, frames, device, batch_size, external_root
        )
    )
    secondary_features = normalize_rows(
        extract_named_frame_model(
            secondary, samples, frames, device, batch_size, external_root
        )
    )
    sweep: dict[str, Any] = {}
    for secondary_weight in fusion_weights:
        if not 0.0 <= secondary_weight <= 1.0:
            raise ValueError("Fusion weights must be between zero and one")
        fused = np.concatenate(
            [
                math.sqrt(1.0 - secondary_weight) * primary_features,
                math.sqrt(secondary_weight) * secondary_features,
            ],
            axis=1,
        )
        sweep[f"{secondary_weight:.2f}"] = evaluate_frame_features(fused, samples)
    selected_weight, selected_metrics = max(
        sweep.items(),
        key=lambda item: item[1]["tracker_to_visible_retrieval"]["rank1_pct"],
    )
    return (
        {
            **selected_metrics,
            "selected_secondary_weight": float(selected_weight),
            "selection_metric": "tracker_to_visible_retrieval.rank1_pct",
            "fusion_sweep": sweep,
        },
        int(primary_features.shape[1] + secondary_features.shape[1]),
    )


def model_display_name(
    model: str,
    weights: Path | None,
    primary: str | None = None,
    secondary: str | None = None,
) -> str:
    if model == "boxmot":
        return weights.stem if weights else "boxmot"
    if model == "fusion":
        return f"{primary} + {secondary}"
    return {
        "hsv": "HSV histogram",
        "dinov2": "DINOv2-small",
        "transreid": "TransReID ViT-B MSMT17",
        "solider": "SOLIDER Swin-S MSMT17",
        "opengait": "OpenGait GaitBase Gait3D",
    }[model]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "hsv",
            "boxmot",
            "dinov2",
            "transreid",
            "solider",
            "opengait",
            "fusion",
        ],
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--primary",
        choices=["clip", "transreid", "osnet_ain"],
        default="clip",
    )
    parser.add_argument(
        "--secondary",
        choices=["transreid", "solider", "dinov2", "osnet_ain", "opengait"],
    )
    parser.add_argument(
        "--fusion-weights",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
    )
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    samples, frames = load_dataset(args.gt_dir)
    if args.model == "opengait":
        metrics = evaluate_opengait(samples, frames, args.device, args.external_root)
        feature_dimension = None
    elif args.model == "fusion":
        if args.secondary is None:
            parser.error("--secondary is required for --model fusion")
        metrics, feature_dimension = evaluate_fusion(
            args.primary,
            args.secondary,
            args.fusion_weights,
            samples,
            frames,
            args.device,
            args.batch_size,
            args.external_root,
        )
    else:
        if args.model == "hsv":
            features = extract_hsv(samples)
        elif args.model == "boxmot":
            if args.weights is None:
                parser.error("--weights is required for --model boxmot")
            features = extract_boxmot(
                samples, frames, args.weights.resolve(), args.device
            )
        elif args.model == "dinov2":
            features = extract_dinov2(samples, args.device, args.batch_size)
        elif args.model == "transreid":
            features = extract_transreid(
                samples, args.device, args.batch_size, args.external_root
            )
        elif args.model == "solider":
            features = extract_solider(
                samples, args.device, args.batch_size, args.external_root
            )
        else:
            raise AssertionError(args.model)
        feature_dimension = int(features.shape[1])
        metrics = evaluate_frame_features(features, samples)

    result = {
        "model": args.model,
        "display_name": model_display_name(
            args.model, args.weights, args.primary, args.secondary
        ),
        "primary": args.primary if args.model == "fusion" else None,
        "secondary": args.secondary,
        "weights": str(args.weights.resolve()) if args.weights else None,
        "protocol": "groundTruth_AllTracking_ARG_FRA_183303 annotated crops",
        "sample_count": len(samples),
        "identity_count": len({sample["identity"] for sample in samples}),
        "visible_crop_count": sum(sample["visible"] for sample in samples),
        "tracker_only_crop_count": sum(not sample["visible"] for sample in samples),
        "feature_dimension": feature_dimension,
        "runtime_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
