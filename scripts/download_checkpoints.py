#!/usr/bin/env python3
"""Restore the public model checkpoints expected by the project configs."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GoogleDriveCheckpoint:
    path: str
    file_id: str
    sha256: str
    source: str


YOLO_CHECKPOINTS = {
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "yolo11s.pt": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    "yolo11m.pt": "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
    "yolo11m-seg.pt": "eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf",
    "yolo11x-seg.pt": "4e53a5f5fd3ee2ae3361c62169c6bb3ed4ae251dd0e57606e230955aa52d919c",
    "yolo11m-pose.pt": "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2",
}

GDRIVE_CHECKPOINTS = (
    GoogleDriveCheckpoint(
        "clip_market1501.pt",
        "1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7",
        "bedc8fcc37f296045df19d901b162dd032bfa85d9b4f2406eff8313536d444dc",
        "BoxMOT ReID model zoo",
    ),
    GoogleDriveCheckpoint(
        "osnet_x0_25_msmt17.pt",
        "1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
        "6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99",
        "BoxMOT ReID model zoo",
    ),
    GoogleDriveCheckpoint(
        "osnet_x1_0_msmt17.pt",
        "112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
        "b7d73dc67c016fd044e4027ff856019496392a7aca8fa0ed56d862a1632c1cf2",
        "BoxMOT ReID model zoo",
    ),
    GoogleDriveCheckpoint(
        "osnet_ain_x1_0_msmt17.pt",
        "1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
        "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
        "BoxMOT ReID model zoo",
    ),
    GoogleDriveCheckpoint(
        "models/parseq_soccernet.ckpt",
        "1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
        "14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879",
        "mkoshkina/jersey-number-pipeline SoccerNet PARSeq",
    ),
    GoogleDriveCheckpoint(
        "models/mkoshkina_legibility_resnet34.pth",
        "18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw",
        "b9c61dabaea4a6ec99528c5ae394f5875aecb8207de38484eccb0f977a373e41",
        "mkoshkina/jersey-number-pipeline SoccerNet legibility classifier",
    ),
    GoogleDriveCheckpoint(
        "models/mixsort/MixFormer_soccernet_train.pth.tar",
        "1FjH4mVdDyRuRJM5aHgYWeOnNjW3x-SLI",
        "c5cae9f8881545a3049505e0538543adf2fb97ddd295bb8ebdb6bd3605d432c2",
        "MCG-NJU/MixSort model zoo",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_ready(path: Path, expected_sha256: str) -> bool:
    if not path.is_file():
        return False
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {path.relative_to(ROOT)}: "
            f"expected {expected_sha256}, got {actual}. "
            "Delete the file and run this script again."
        )
    print(f"[ok] {path.relative_to(ROOT)}")
    return True


def download_yolo_checkpoints() -> None:
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics first: pip install ultralytics") from exc

    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for name, expected_sha256 in YOLO_CHECKPOINTS.items():
            destination = ROOT / name
            if is_ready(destination, expected_sha256):
                continue
            print(f"[download] {name} from Ultralytics release assets")
            downloaded = Path(attempt_download_asset(name)).resolve()
            if downloaded != destination.resolve():
                shutil.move(str(downloaded), destination)
            is_ready(destination, expected_sha256)
    finally:
        os.chdir(previous_cwd)


def download_gdrive_checkpoints() -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown first: pip install gdown") from exc

    for checkpoint in GDRIVE_CHECKPOINTS:
        destination = ROOT / checkpoint.path
        if is_ready(destination, checkpoint.sha256):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {checkpoint.path} from {checkpoint.source}")
        result = gdown.download(
            id=checkpoint.file_id,
            output=str(destination),
            quiet=False,
        )
        if result is None:
            raise RuntimeError(f"Download failed for {checkpoint.path}")
        is_ready(destination, checkpoint.sha256)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the public checkpoints used by sports_reID."
    )
    parser.add_argument(
        "--group",
        choices=("all", "yolo", "reid-ocr"),
        default="all",
        help="Limit downloads to one checkpoint group.",
    )
    args = parser.parse_args()

    if args.group in {"all", "yolo"}:
        download_yolo_checkpoints()
    if args.group in {"all", "reid-ocr"}:
        download_gdrive_checkpoints()

    print("Checkpoint setup complete.")


if __name__ == "__main__":
    main()
