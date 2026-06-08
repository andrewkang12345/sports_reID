from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class TrackletCandidateDataset:
    """Dataset for roster-conditioned fusion training.

    Expected JSONL row:
    {
      "features": [float, ...],
      "label": 0 or 1,
      "track_id": "...",
      "candidate_player_id": "..."
    }
    """

    def __init__(self, path: str | Path) -> None:
        self.rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"No rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        row = self.rows[index]
        return np.asarray(row["features"], dtype=np.float32), int(row["label"])

    @property
    def feature_dim(self) -> int:
        return int(len(self.rows[0]["features"]))
