#!/usr/bin/env python3
"""Small atomic FIFO used by ROS Python 3.10 and inference Python 3.11."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


FRAME_PREFIX = "frame_"
FRAME_SUFFIX = ".npz"


def queue_dir(ipc: Path) -> Path:
    path = ipc / "frames"
    path.mkdir(parents=True, exist_ok=True)
    return path


def frame_files(ipc: Path) -> list[Path]:
    return sorted(queue_dir(ipc).glob(f"{FRAME_PREFIX}*{FRAME_SUFFIX}"))


def clear_frame_queue(ipc: Path) -> int:
    removed = 0
    for path in queue_dir(ipc).glob(f"{FRAME_PREFIX}*"):
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    return removed


def frame_sequence(path: Path) -> int:
    return int(path.stem.removeprefix(FRAME_PREFIX))


def enqueue_frame(
    ipc: Path,
    sequence: int,
    metadata: dict,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
) -> Path:
    """Atomically append one RGB-D frame to the queue without pickling."""
    directory = queue_dir(ipc)
    final = directory / f"{FRAME_PREFIX}{sequence:012d}{FRAME_SUFFIX}"
    tmp = directory / f".{final.name}.tmp"
    encoded_meta = np.frombuffer(
        json.dumps(metadata, ensure_ascii=False).encode("utf-8"), dtype=np.uint8
    )
    with tmp.open("wb") as stream:
        np.savez(
            stream,
            color_bgr=np.ascontiguousarray(color_bgr),
            depth_raw=np.ascontiguousarray(depth_raw),
            metadata_utf8=encoded_meta,
        )
    os.replace(tmp, final)
    return final


def dequeue_frame(
    ipc: Path, after_sequence: int = -1
) -> tuple[dict, np.ndarray, np.ndarray, Path] | None:
    """Load and consume the oldest queued frame newer than after_sequence."""
    for path in frame_files(ipc):
        try:
            sequence = frame_sequence(path)
        except ValueError:
            continue
        if sequence <= after_sequence:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        try:
            with np.load(path, allow_pickle=False) as bundle:
                color_bgr = np.asarray(bundle["color_bgr"]).copy()
                depth_raw = np.asarray(bundle["depth_raw"]).copy()
                meta_bytes = np.asarray(bundle["metadata_utf8"], dtype=np.uint8).tobytes()
            metadata = json.loads(meta_bytes.decode("utf-8"))
        except (FileNotFoundError, OSError, ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
            # A damaged queue entry must not keep the bounded FIFO full forever.
            if path.exists():
                path.rename(path.with_suffix(".invalid"))
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        if color_bgr.shape[:2] != depth_raw.shape[:2]:
            continue
        return metadata, color_bgr, depth_raw, path
    return None
