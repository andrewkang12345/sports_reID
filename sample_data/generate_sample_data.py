from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "test_clip.mp4"
METADATA_PATH = ROOT / "test_metadata.json"


PLAYERS = [
    {
        "player_name": "Ola Brynhildsen",
        "team": "Molde FK",
        "jersey": "9",
        "position": "FW",
        "color": (35, 35, 210),  # BGR
        "path": [(170, 335), (260, 330), (360, 318), (455, 315), (545, 320)],
    },
    {
        "player_name": "Emil Breivik",
        "team": "Molde FK",
        "jersey": "16",
        "position": "MF",
        "color": (35, 35, 210),
        "path": [(250, 215), (340, 220), (430, 225), (520, 230), (610, 235)],
    },
    {
        "player_name": "Noah Holm",
        "team": "Rosenborg BK",
        "jersey": "11",
        "position": "FW",
        "color": (215, 75, 30),
        "path": [(735, 285), (665, 288), (595, 292), (525, 296), (455, 300)],
    },
    {
        "player_name": "Markus Henriksen",
        "team": "Rosenborg BK",
        "jersey": "7",
        "position": "MF",
        "color": (215, 75, 30),
        "path": [(700, 390), (635, 380), (570, 370), (505, 360), (440, 350)],
    },
]


def interp_path(points: list[tuple[int, int]], alpha: float) -> tuple[float, float]:
    if alpha <= 0:
        return points[0]
    if alpha >= 1:
        return points[-1]
    scaled = alpha * (len(points) - 1)
    idx = int(np.floor(scaled))
    frac = scaled - idx
    x0, y0 = points[idx]
    x1, y1 = points[idx + 1]
    return x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac


def ball_position(t: float) -> tuple[float, float]:
    if t < 1.2:
        return 125 + 60 * t, 350
    if t < 3.0:
        p = PLAYERS[0]
        x, y = interp_path(p["path"], t / 10.0)
        return x + 14, y + 42
    if t < 4.6:
        alpha = (t - 3.0) / 1.6
        x0, y0 = 395, 348
        x1, y1 = 520, 270
        return x0 + (x1 - x0) * alpha, y0 + (y1 - y0) * alpha
    if t < 6.2:
        p = PLAYERS[1]
        x, y = interp_path(p["path"], t / 10.0)
        return x + 8, y + 40
    if t < 7.6:
        p = PLAYERS[2]
        x, y = interp_path(p["path"], t / 10.0)
        return x - 8, y + 44
    p = PLAYERS[3]
    x, y = interp_path(p["path"], t / 10.0)
    return x + 10, y + 44


def draw_field(frame: np.ndarray) -> None:
    frame[:] = (46, 132, 58)
    h, w = frame.shape[:2]
    for x in range(0, w, 120):
        shade = 8 if (x // 120) % 2 == 0 else -8
        frame[:, x : x + 120] = np.clip(frame[:, x : x + 120].astype(np.int16) + shade, 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (55, 55), (w - 55, h - 55), (230, 230, 230), 2)
    cv2.line(frame, (w // 2, 55), (w // 2, h - 55), (230, 230, 230), 2)
    cv2.circle(frame, (w // 2, h // 2), 70, (230, 230, 230), 2)
    cv2.rectangle(frame, (55, 185), (155, 355), (230, 230, 230), 2)
    cv2.rectangle(frame, (w - 155, 185), (w - 55, 355), (230, 230, 230), 2)


def draw_player(frame: np.ndarray, player: dict, t: float) -> None:
    x, y = interp_path(player["path"], t / 10.0)
    x = int(round(x))
    y = int(round(y))
    jersey_color = player["color"]
    skin = (75, 165, 218)
    shorts = (25, 25, 35)
    cv2.circle(frame, (x, y - 31), 10, skin, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x - 18, y - 20), (x + 18, y + 34), jersey_color, -1)
    cv2.rectangle(frame, (x - 18, y + 28), (x + 18, y + 48), shorts, -1)
    cv2.line(frame, (x - 10, y + 46), (x - 14, y + 70), shorts, 5, cv2.LINE_AA)
    cv2.line(frame, (x + 10, y + 46), (x + 14, y + 70), shorts, 5, cv2.LINE_AA)
    cv2.line(frame, (x - 17, y - 8), (x - 28, y + 18), skin, 4, cv2.LINE_AA)
    cv2.line(frame, (x + 17, y - 8), (x + 28, y + 18), skin, 4, cv2.LINE_AA)
    number = str(player["jersey"])
    font_scale = 0.72 if len(number) == 1 else 0.58
    thickness = 2
    (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(
        frame,
        number,
        (x - tw // 2, y + th // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    width, height, fps, seconds = 960, 540, 15, 10
    writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {VIDEO_PATH}")
    for frame_index in range(fps * seconds):
        t = frame_index / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_field(frame)
        for player in sorted(PLAYERS, key=lambda p: interp_path(p["path"], t / 10.0)[1]):
            draw_player(frame, player, t)
        bx, by = ball_position(t)
        cv2.circle(frame, (int(round(bx)), int(round(by))), 7, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(round(bx)), int(round(by))), 7, (20, 20, 20), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    metadata = {
        "home_team": "Molde FK",
        "away_team": "Rosenborg BK",
        "team_colors": {
            "Molde FK": "#d22323",
            "Rosenborg BK": "#1e4bd7",
        },
        "rosters": {
            "Molde FK": [
                {"player_name": "Ola Brynhildsen", "jersey_number": 9, "position": "FW"},
                {"player_name": "Emil Breivik", "jersey_number": 16, "position": "MF"},
            ],
            "Rosenborg BK": [
                {"player_name": "Noah Holm", "jersey_number": 11, "position": "FW"},
                {"player_name": "Markus Henriksen", "jersey_number": 7, "position": "MF"},
            ],
        },
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(VIDEO_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()
