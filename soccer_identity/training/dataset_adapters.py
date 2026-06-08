from __future__ import annotations

import csv
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class MOTBox:
    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int | None = None
    visibility: float | None = None


@dataclass
class MOTSequence:
    name: str
    split: str
    root: Path
    frame_rate: float | None = None
    seq_length: int | None = None
    im_width: int | None = None
    im_height: int | None = None
    im_ext: str = ".jpg"

    @property
    def image_dir(self) -> Path:
        return self.root / "img1"

    @property
    def gt_path(self) -> Path:
        return self.root / "gt" / "gt.txt"

    def iter_boxes(self) -> Iterator[MOTBox]:
        if not self.gt_path.exists():
            return
        with self.gt_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 7:
                    continue
                yield MOTBox(
                    frame=int(float(row[0])),
                    track_id=int(float(row[1])),
                    x=float(row[2]),
                    y=float(row[3]),
                    width=float(row[4]),
                    height=float(row[5]),
                    confidence=float(row[6]),
                    class_id=int(float(row[7])) if len(row) > 7 and row[7] else None,
                    visibility=float(row[8]) if len(row) > 8 and row[8] else None,
                )


def load_mot_sequence(path: str | Path, split: str = "train") -> MOTSequence:
    root = Path(path)
    seq = MOTSequence(name=root.name, split=split, root=root)
    seqinfo = root / "seqinfo.ini"
    if seqinfo.exists():
        parser = configparser.ConfigParser()
        parser.read(seqinfo)
        section = parser["Sequence"] if parser.has_section("Sequence") else {}
        seq.frame_rate = float(section.get("frameRate", 0) or 0) or None
        seq.seq_length = int(section.get("seqLength", 0) or 0) or None
        seq.im_width = int(section.get("imWidth", 0) or 0) or None
        seq.im_height = int(section.get("imHeight", 0) or 0) or None
        seq.im_ext = section.get("imExt", ".jpg") or ".jpg"
    return seq


def discover_sportsmot_sequences(root: str | Path) -> list[MOTSequence]:
    dataset_root = Path(root)
    if (dataset_root / "dataset").exists():
        dataset_root = dataset_root / "dataset"
    sequences: list[MOTSequence] = []
    for split in ("train", "val", "test"):
        split_root = dataset_root / split
        if not split_root.exists():
            continue
        for seq_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            if (seq_dir / "img1").exists():
                sequences.append(load_mot_sequence(seq_dir, split=split))
    return sequences


def summarize_mot_sequences(sequences: list[MOTSequence]) -> dict[str, float]:
    total_boxes = 0
    total_tracks: set[tuple[str, int]] = set()
    total_frames = 0
    for seq in sequences:
        if seq.seq_length:
            total_frames += seq.seq_length
        for box in seq.iter_boxes():
            total_boxes += 1
            total_tracks.add((seq.name, box.track_id))
    return {
        "num_sequences": float(len(sequences)),
        "num_frames": float(total_frames),
        "num_boxes": float(total_boxes),
        "num_tracks": float(len(total_tracks)),
    }
