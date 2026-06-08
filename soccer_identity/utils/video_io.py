from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float


def get_video_info(path: str | Path) -> VideoInfo:
    video_path = str(path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 25.0
    return VideoInfo(
        path=video_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration=frame_count / fps if frame_count else 0.0,
    )


def iter_video_frames(
    path: str | Path,
    max_seconds: float | None = None,
    stride: int = 1,
) -> Iterator[tuple[int, float, np.ndarray]]:
    info = get_video_info(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    max_frames = None
    if max_seconds is not None and max_seconds > 0:
        max_frames = int(round(max_seconds * info.fps))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_index >= max_frames:
            break
        if stride <= 1 or frame_index % stride == 0:
            yield frame_index, frame_index / info.fps, frame
        frame_index += 1
    cap.release()


def create_video_writer(path: str | Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_name in ("mp4v", "avc1", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open video writer for {output_path}")


def transcode_to_browser_mp4(input_path: str | Path, output_path: str | Path | None = None, crf: int = 22) -> bool:
    """Transcode to H.264/yuv420p MP4, which VS Code and browsers preview reliably."""
    src = Path(input_path)
    dst = Path(output_path) if output_path is not None else src
    if not src.exists() or src.stat().st_size == 0:
        return False
    try:
        import imageio_ffmpeg
    except Exception:
        return False

    tmp = dst.with_name(f"{dst.stem}.h264_tmp{dst.suffix}")
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        return False
    tmp.replace(dst)
    return dst.exists() and dst.stat().st_size > 0
