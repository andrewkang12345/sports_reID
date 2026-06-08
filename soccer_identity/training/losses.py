from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_identity_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1))


def auxiliary_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return float(weight) * F.cross_entropy(logits, labels.long())


def temporal_consistency_loss(left_logits: torch.Tensor, right_logits: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    left = torch.log_softmax(left_logits, dim=-1)
    right = torch.softmax(right_logits.detach(), dim=-1)
    return float(weight) * F.kl_div(left, right, reduction="batchmean")
