from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Interval:
    label: str
    start_time: float
    end_time: float


def interval_iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a.end_time, b.end_time) - max(a.start_time, b.start_time))
    union = max(a.end_time, b.end_time) - min(a.start_time, b.start_time)
    return inter / union if union > 0 else 0.0


def near_ball_event_f1(predicted: list[Interval], ground_truth: list[Interval], iou_threshold: float = 0.3) -> dict[str, float]:
    used: set[int] = set()
    tp = 0
    for pred in predicted:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(ground_truth):
            if idx in used or pred.label != gt.label:
                continue
            score = interval_iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            used.add(best_idx)
            tp += 1
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}
