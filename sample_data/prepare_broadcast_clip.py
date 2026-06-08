from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "test_clip.mp4"
METADATA_PATH = ROOT / "test_metadata.json"

DEFAULT_SOURCE = Path(
    "/mnt/data/mywork/fifaInnovationChallenge/starter_kit/data/videos/BRA_KOR_230503.mp4"
)

TEAM_NAMES = {
    "ARG": "Argentina",
    "BRA": "Brazil",
    "CRO": "Croatia",
    "ENG": "England",
    "FRA": "France",
    "KOR": "South Korea",
    "MOR": "Morocco",
    "NET": "Netherlands",
    "POR": "Portugal",
}

TEAM_COLORS = {
    "Argentina": {"shirt": "#75aadb", "shorts": "#ffffff"},
    "Brazil": {"shirt": "#f7e017", "shorts": "#1d4aa8"},
    "Croatia": {"shirt": "#ffffff", "shorts": "#ffffff"},
    "England": {"shirt": "#ffffff", "shorts": "#ffffff"},
    "France": {"shirt": "#1b2f5f", "shorts": "#1b2f5f"},
    "Morocco": {"shirt": "#c1272d", "shorts": "#1f8f45"},
    "Netherlands": {"shirt": "#f26b21", "shorts": "#f26b21"},
    "Portugal": {"shirt": "#c1272d", "shorts": "#1f8f45"},
    "South Korea": {"shirt": "#d71920", "shorts": "#111111"},
}


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


# Real squads for clips where we have public match metadata. Used when invoked with
# --use_ground_truth_roster. Source: 2022 FIFA World Cup Final, 18 Dec 2022.
WC_2022_ARGENTINA_SQUAD: list[tuple[int, str, str]] = [
    (1, "Franco Armani", "GK"),
    (2, "Lisandro Martínez", "DF"),
    (3, "Nicolás Tagliafico", "DF"),
    (4, "Gonzalo Montiel", "DF"),
    (5, "Leandro Paredes", "MF"),
    (6, "Germán Pezzella", "DF"),
    (7, "Rodrigo De Paul", "MF"),
    (8, "Marcos Acuña", "DF"),
    (9, "Julián Álvarez", "FW"),
    (10, "Lionel Messi", "FW"),
    (11, "Ángel Di María", "FW"),
    (12, "Gerónimo Rulli", "GK"),
    (13, "Cristian Romero", "DF"),
    (14, "Exequiel Palacios", "MF"),
    (15, "Ángel Correa", "FW"),
    (16, "Thiago Almada", "MF"),
    (17, "Alejandro Gómez", "FW"),
    (18, "Guido Rodríguez", "MF"),
    (19, "Nicolás Otamendi", "DF"),
    (20, "Alexis Mac Allister", "MF"),
    (21, "Paulo Dybala", "FW"),
    (22, "Lautaro Martínez", "FW"),
    (23, "Emiliano Martínez", "GK"),
    (24, "Enzo Fernández", "MF"),
    (25, "Juan Foyth", "DF"),
    (26, "Nahuel Molina", "DF"),
]

WC_2022_FRANCE_SQUAD: list[tuple[int, str, str]] = [
    (1, "Hugo Lloris", "GK"),
    (2, "Benjamin Pavard", "DF"),
    (3, "Axel Disasi", "DF"),
    (4, "Raphaël Varane", "DF"),
    (5, "Jules Koundé", "DF"),
    (6, "Eduardo Camavinga", "MF"),
    (7, "Antoine Griezmann", "MF"),
    (8, "Aurélien Tchouaméni", "MF"),
    (9, "Olivier Giroud", "FW"),
    (10, "Kylian Mbappé", "FW"),
    (11, "Ousmane Dembélé", "FW"),
    (12, "Randal Kolo Muani", "FW"),
    (13, "Youssouf Fofana", "MF"),
    (14, "Adrien Rabiot", "MF"),
    (15, "Jordan Veretout", "MF"),
    (16, "Steve Mandanda", "GK"),
    (17, "William Saliba", "DF"),
    (18, "Dayot Upamecano", "DF"),
    (19, "Karim Benzema", "FW"),
    (20, "Kingsley Coman", "FW"),
    (21, "Lucas Hernández", "DF"),
    (22, "Théo Hernández", "DF"),
    (23, "Marcus Thuram", "FW"),
    (24, "Ibrahima Konaté", "DF"),
    (25, "Eduardo Camavinga", "MF"),
    (26, "Matteo Guendouzi", "MF"),
]


# Ground-truth-ish rosters for clips where the user has confirmed visible jersey numbers.
# When set to a full-squad string key like "wc_2022_argentina", we load the full 26-player
# squad with real names; numeric list still works as a fallback (just numbers, generic names).
GROUND_TRUTH_ROSTERS: dict[str, dict[str, object]] = {
    "ARG_FRA_183303": {
        "Argentina": "wc_2022_argentina",
        "France": "wc_2022_france",
    },
}

# Numbers actually visible on screen in the clip. Used to constrain OCR candidates while
# still letting the full squad provide real player names for the visualization.
VISIBLE_JERSEY_NUMBERS: dict[str, dict[str, list[int]]] = {
    "ARG_FRA_183303": {
        # Source: user-confirmed ground truth from outputs/ground_truth_arg_fra.json.
        "Argentina": [9, 7, 10, 19, 20, 24, 26],
        "France":    [4, 7, 9, 10, 14, 18, 22],
    },
}

FULL_SQUADS: dict[str, list[tuple[int, str, str]]] = {
    "wc_2022_argentina": WC_2022_ARGENTINA_SQUAD,
    "wc_2022_france": WC_2022_FRANCE_SQUAD,
}


def roster_from_numbers(team_name: str, numbers: list[int]) -> list[dict[str, object]]:
    return [
        {
            "player_name": f"{team_name} #{num}",
            "team_name": team_name,
            "jersey_number": num,
            "position": "?",
        }
        for num in numbers
    ]


def roster_from_squad(team_name: str, squad: list[tuple[int, str, str]]) -> list[dict[str, object]]:
    return [
        {
            "player_name": name,
            "team_name": team_name,
            "jersey_number": num,
            "position": pos,
        }
        for num, name, pos in squad
    ]


def transcode(source: Path, start_sec: float, duration_sec: float) -> None:
    if not source.exists():
        raise FileNotFoundError(
            f"Broadcast source clip not found: {source}. "
            "Provide a local broadcast soccer MP4 via --source."
        )
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_sec:.3f}",
        "-an",
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
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


def teams_from_source(source: Path) -> tuple[str, str]:
    parts = source.stem.split("_")
    if len(parts) >= 2:
        return TEAM_NAMES.get(parts[0], parts[0]), TEAM_NAMES.get(parts[1], parts[1])
    return "Home Team", "Away Team"


def _resolve_roster(team_name: str, entry: object) -> list[dict[str, object]]:
    if isinstance(entry, str) and entry in FULL_SQUADS:
        return roster_from_squad(team_name, FULL_SQUADS[entry])
    if isinstance(entry, list):
        return roster_from_numbers(team_name, [int(x) for x in entry])  # type: ignore[arg-type]
    return anonymous_roster(team_name, team_name)


def write_metadata(source: Path, start_sec: float, duration_sec: float, use_ground_truth: bool = False) -> None:
    home_team, away_team = teams_from_source(source)
    gt = GROUND_TRUTH_ROSTERS.get(source.stem, {}) if use_ground_truth else {}
    home_roster = _resolve_roster(home_team, gt[home_team]) if home_team in gt else anonymous_roster(home_team, home_team)
    away_roster = _resolve_roster(away_team, gt[away_team]) if away_team in gt else anonymous_roster(away_team, away_team)
    # When real player names are available, enable identity-label rendering so the visualization
    # actually says "Messi #10" rather than "Low conf ID #10".
    has_real_names = bool(gt)
    visible_lookup = VISIBLE_JERSEY_NUMBERS.get(source.stem, {}) if use_ground_truth else {}
    visible_numbers = {team: list(nums) for team, nums in visible_lookup.items()} if visible_lookup else {}
    metadata = {
        "home_team": home_team,
        "away_team": away_team,
        "identity_labels_available": has_real_names,
        "visible_jersey_numbers": visible_numbers,
        "team_colors": {
            home_team: TEAM_COLORS.get(home_team, {"shirt": "#ffffff"}),
            away_team: TEAM_COLORS.get(away_team, {"shirt": "#000000"}),
        },
        "public_source": {
            "dataset": "FIFA Skeletal Tracking Light 2026",
            "dataset_url": "https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026/tree/main",
            "source_clip": str(source),
            "source_sequence": source.stem,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "license_note": "Dataset page lists license cc-by-2.0 and gated non-commercial access terms.",
            "roster_source": "ground_truth_user_provided" if (home_team in gt or away_team in gt) else "anonymous_1_to_11",
        },
        "rosters": {
            home_team: home_roster,
            away_team: away_roster,
        },
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a 10-second broadcast-style soccer test clip.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--use_ground_truth_roster", action="store_true",
                        help="When set, use GROUND_TRUTH_ROSTERS for clips that have one configured.")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    transcode(source, args.start_sec, args.duration_sec)
    write_metadata(source, args.start_sec, args.duration_sec, use_ground_truth=args.use_ground_truth_roster)
    print(VIDEO_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()
