#!/usr/bin/env python3
"""Download checkpoints used only by archived experiments."""

from __future__ import annotations

from pathlib import Path


CHECKPOINTS = (
    ("osnet_x0_25_msmt17.pt", "1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"),
    ("osnet_ain_x1_0_msmt17.pt", "1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal"),
    (
        "models/mixsort/MixFormer_soccernet_train.pth.tar",
        "1FjH4mVdDyRuRJM5aHgYWeOnNjW3x-SLI",
    ),
)


def main() -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown first: pip install gdown") from exc

    root = Path(__file__).resolve().parents[2]
    for relative_path, file_id in CHECKPOINTS:
        destination = root / relative_path
        if destination.is_file():
            print(f"[ok] {relative_path}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {relative_path}")
        if gdown.download(id=file_id, output=str(destination), quiet=False) is None:
            raise RuntimeError(f"Download failed for {relative_path}")


if __name__ == "__main__":
    main()
