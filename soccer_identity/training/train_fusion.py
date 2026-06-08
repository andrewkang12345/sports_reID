from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from soccer_identity.training.dataset import TrackletCandidateDataset
from soccer_identity.training.losses import binary_identity_loss


class FusionMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(-1)


@dataclass
class TrainConfig:
    train_jsonl: str
    output_path: str
    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    hidden_dim: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_fusion_model(config: TrainConfig) -> dict[str, float]:
    dataset = TrackletCandidateDataset(config.train_jsonl)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    model = FusionMLP(dataset.feature_dim, config.hidden_dim).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    final_loss = 0.0
    for _epoch in range(config.epochs):
        model.train()
        running = 0.0
        count = 0
        for features, labels in loader:
            features = features.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = binary_identity_loss(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(features.shape[0])
            count += int(features.shape[0])
        final_loss = running / max(1, count)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": dataset.feature_dim,
            "hidden_dim": config.hidden_dim,
        },
        output_path,
    )
    return {"train_loss": final_loss, "feature_dim": float(dataset.feature_dim)}
