from __future__ import annotations

import argparse
from pathlib import Path

from soccer_identity.training.dataset_adapters import discover_sportsmot_sequences, summarize_mot_sequences
from soccer_identity.utils.schemas import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate production dataset roots and write a training manifest.")
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--sportsmot_root", default=None, help="SportsMOT root in MOTChallenge layout.")
    parser.add_argument("--soccernet_gsr_root", default=None)
    parser.add_argument("--soccernet_reid_root", default=None)
    parser.add_argument("--soccernet_jersey_root", default=None)
    parser.add_argument("--soccertrack_root", default=None)
    return parser.parse_args()


def path_status(path: str | None) -> dict[str, object]:
    if not path:
        return {"configured": False, "exists": False, "path": None}
    p = Path(path)
    return {"configured": True, "exists": p.exists(), "path": str(p)}


def main() -> None:
    args = parse_args()
    manifest = {
        "datasets": {
            "sportsmot": path_status(args.sportsmot_root),
            "soccernet_gsr": path_status(args.soccernet_gsr_root),
            "soccernet_reid": path_status(args.soccernet_reid_root),
            "soccernet_jersey": path_status(args.soccernet_jersey_root),
            "soccertrack_v2": path_status(args.soccertrack_root),
        },
        "sportsmot_summary": {},
        "next_steps": [
            "Train detector/tracker substrate on SportsMOT/TeamTrack/SoccerTrack MOT annotations.",
            "Train jersey OCR on SoccerNet Jersey Number and optional hockey jersey datasets.",
            "Train/fine-tune body ReID on SoccerNet ReID and sports crop tracklets.",
            "Export frozen tracklet evidence into fusion JSONL rows.",
            "Train train_identity_fusion.py on roster-conditioned candidate rows.",
        ],
    }
    if args.sportsmot_root and Path(args.sportsmot_root).exists():
        sequences = discover_sportsmot_sequences(args.sportsmot_root)
        manifest["sportsmot_summary"] = summarize_mot_sequences(sequences)
    write_json(args.output_manifest, manifest)
    print(args.output_manifest)


if __name__ == "__main__":
    main()
