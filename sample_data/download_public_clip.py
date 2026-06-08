from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "juvisy_guingamp_public_source.webm"
VIDEO_PATH = ROOT / "test_clip.mp4"
METADATA_PATH = ROOT / "test_metadata.json"

SOURCE_URL = (
    "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
    "Juvisy%20Guingamp%20Breuillet%2017%20Aout%202013%20-%2016%20-%20Premier%20but%20Juvisy.webm"
)

# Fallback direct URL for Wikimedia's canonical file path. Kept separate because
# Special:Redirect handles most future storage moves, but the file title contains
# spaces and apostrophes that can be brittle in non-browser clients.
DIRECT_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/d/da/"
    "Juvisy_Guingamp_Breuillet_17_Aout_2013_-_16_-_Premier_but_Juvisy.webm"
)


def anonymous_roster(team_name: str, prefix: str) -> list[dict[str, object]]:
    positions = ["GK", "DF", "DF", "DF", "DF", "MF", "MF", "MF", "FW", "FW", "FW"]
    return [
        {
            "player_name": f"{prefix} roster slot {idx}",
            "team_name": team_name,
            "jersey_number": idx,
            "position": positions[idx - 1],
        }
        for idx in range(1, 12)
    ]


def download() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists() or RAW_PATH.stat().st_size == 0:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "sports-reid-demo/0.1 (public sample clip downloader)")]
        try:
            with opener.open(SOURCE_URL) as response, RAW_PATH.open("wb") as f:
                f.write(response.read())
        except Exception:
            with opener.open(DIRECT_URL) as response, RAW_PATH.open("wb") as f:
                f.write(response.read())


def transcode() -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        "0",
        "-i",
        str(RAW_PATH),
        "-t",
        "10",
        "-an",
        "-vf",
        "scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(VIDEO_PATH),
    ]
    subprocess.run(cmd, check=True)


def write_metadata() -> None:
    metadata = {
        "home_team": "FCF Juvisy Essonne",
        "away_team": "EA Guingamp",
        "public_source": {
            "title": "Juvisy Guingamp Breuillet 17 Aout 2013 - 16 - Premier but Juvisy.webm",
            "page_url": "https://commons.wikimedia.org/wiki/File:Juvisy_Guingamp_Breuillet_17_Aout_2013_-_16_-_Premier_but_Juvisy.webm",
            "license": "CC0 1.0 Universal Public Domain Dedication",
            "author": "Shev123",
            "date": "2013-08-17",
        },
        "identity_labels_available": False,
        "rosters": {
            "FCF Juvisy Essonne": anonymous_roster("FCF Juvisy Essonne", "Juvisy"),
            "EA Guingamp": anonymous_roster("EA Guingamp", "Guingamp"),
        },
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def main() -> None:
    download()
    transcode()
    write_metadata()
    print(VIDEO_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()
