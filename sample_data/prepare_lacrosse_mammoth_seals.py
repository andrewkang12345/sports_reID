from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccer_identity.utils.video_io import create_video_writer, transcode_to_browser_mp4


def _anonymous_roster(team_name: str, prefix: str) -> list[dict[str, object]]:
    # Keep all two-digit candidates available to template OCR without pretending we
    # have real player identities for this broadcast clip.
    return [
        {
            "player_id": f"{prefix}_{num:02d}",
            "player_name": f"{team_name} #{num}",
            "team_name": team_name,
            "jersey_number": num,
            "position": "Runner",
        }
        for num in range(0, 100)
    ]


def write_clip(source: Path, output: Path, start_sec: float, duration_sec: float) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open source video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(round(start_sec * fps))
    max_frames = int(round(duration_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tmp = output.with_name(f"{output.stem}.opencv_tmp{output.suffix}")
    writer = create_video_writer(tmp, fps, (width, height))
    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    if written == 0:
        raise RuntimeError("no frames were written")
    if not transcode_to_browser_mp4(tmp, output, crf=22):
        tmp.replace(output)
    else:
        tmp.unlink(missing_ok=True)


def write_metadata(path: Path, clip_path: Path, source_path: Path, start_sec: float, duration_sec: float) -> None:
    metadata = {
        "sport": "lacrosse",
        "competition": "National Lacrosse League",
        "match_date": "2019-04-19",
        "match_title": "Colorado Mammoth vs. San Diego Seals",
        "venue": "Pechanga Arena",
        "home_team": "San Diego Seals",
        "away_team": "Colorado Mammoth",
        "final_score": {
            "San Diego Seals": 12,
            "Colorado Mammoth": 7,
        },
        "clip": {
            "path": str(clip_path),
            "source_path": str(source_path),
            "start_sec": start_sec,
            "duration_sec": duration_sec,
        },
        "identity_labels_available": False,
        "team_colors": {
            "San Diego Seals": {
                "shirt": "#f4f4f0",
                "shorts": "#111111",
                "helmet": "#f4f4f0",
            },
            "Colorado Mammoth": {
                "shirt": "#15120f",
                "shorts": "#15120f",
                "accent": "#b99a45",
                "helmet": "#15120f",
            },
        },
        "officials": {
            "referee_uniform": {
                "shirt": "black-white vertical stripes",
                "shorts": "#111111",
                "helmet": False,
            }
        },
        "public_source": {
            "video_title": "Colorado Mammoth vs. San Diego Seals 4/19/19 | Full Game",
            "url": "https://www.youtube.com/watch?v=culaKSsITCc",
            "score_source_note": "NLL YouTube listing reports San Diego Seals 12, Colorado Mammoth 7.",
        },
        "rosters": {
            "San Diego Seals": _anonymous_roster("San Diego Seals", "SD"),
            "Colorado Mammoth": _anonymous_roster("Colorado Mammoth", "COL"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a 30s Colorado Mammoth vs San Diego Seals lacrosse demo clip.")
    parser.add_argument("--source", type=Path, default=Path("/mnt/data/lacrosse_long_source_iniyaa.mp4"))
    parser.add_argument("--output-video", type=Path, default=Path("sample_data/lacrosse_mammoth_seals_2019_30s.mp4"))
    parser.add_argument("--metadata", type=Path, default=Path("sample_data/lacrosse_mammoth_seals_2019_metadata.json"))
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    args = parser.parse_args()

    write_clip(args.source, args.output_video, args.start_sec, args.duration_sec)
    write_metadata(args.metadata, args.output_video, args.source, args.start_sec, args.duration_sec)
    print(f"wrote {args.output_video}")
    print(f"wrote {args.metadata}")


if __name__ == "__main__":
    main()
