"""Helpers for loading packed RGB/depth NPZ bundles.

The datasets bundled under ``packed_npz`` contain per-camera RGB frames,
depth maps, intrinsics, and extrinsics stored as numpy arrays.  This module
converts those bundles into the ``views`` structure expected by
``MapAnything.infer`` after running them through ``preprocess_inputs`` for
normalisation and resizing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapanything.utils.image import preprocess_inputs


def _load_npz(npz_path: Path) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def summarize_packed_npz(npz_path: Path) -> Dict[str, Any]:
    """Return a lightweight JSON-serialisable description of an NPZ bundle."""

    contents = _load_npz(npz_path)

    camera_ids = [str(x) for x in contents["camera_ids"]]
    timestamps = contents.get("timestamps")

    summary: Dict[str, Any] = {
        "path": str(npz_path),
        "episode": str(contents.get("episode", "")),
        "camera_count": len(camera_ids),
        "frame_count": int(contents["rgbs"].shape[1]) if contents["rgbs"].ndim >= 2 else 0,
        "camera_ids": camera_ids,
        "rgb_shape": list(contents["rgbs"].shape),
        "depth_shape": list(contents["depths"].shape),
        "intrinsics_shape": list(contents["intrs"].shape),
        "extrinsics_shape": list(contents["extrs"].shape),
    }

    if timestamps is not None:
        summary["timestamp_range"] = [int(timestamps.min()), int(timestamps.max())]

    return summary


def _resolve_camera_indices(
    selectors: Optional[Sequence[str]], camera_ids: Sequence[str]
) -> List[int]:
    if not selectors:
        return list(range(len(camera_ids)))

    resolved: List[int] = []
    for item in selectors:
        if item.isdigit():
            idx = int(item)
            if idx < 0 or idx >= len(camera_ids):
                raise ValueError(f"Camera index {idx} is out of range (0-{len(camera_ids) - 1})")
            resolved.append(idx)
            continue

        if item not in camera_ids:
            raise ValueError(
                f"Unknown camera selector '{item}'. Available ids: {', '.join(camera_ids)}"
            )
        resolved.append(camera_ids.index(item))

    return resolved


def _resolve_frame_indices(
    selectors: Optional[Sequence[str]], frame_count: int
) -> List[int]:
    if not selectors:
        return list(range(frame_count))

    frames: List[int] = []
    for item in selectors:
        try:
            idx = int(item)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Frame selector '{item}' is not an integer."
            ) from exc

        if idx < 0 or idx >= frame_count:
            raise ValueError(f"Frame index {idx} is out of range (0-{frame_count - 1})")
        frames.append(idx)

    return frames


def _pose_3x4_to_4x4(pose_3x4: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :4] = pose_3x4.astype(np.float32)
    return pose


def views_from_packed_npz(
    npz_path: Path,
    cameras: Optional[Sequence[str]] = None,
    frames: Optional[Sequence[str]] = None,
    resize_mode: str = "fixed_mapping",
    size: Optional[Any] = None,
    norm_type: str = "dinov2",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert a packed NPZ file into MapAnything view dictionaries."""

    contents = _load_npz(npz_path)

    rgbs = contents["rgbs"]  # (C, F, 3, H, W)
    depths = contents["depths"]  # (C, F, 1, H, W)
    intrs = contents["intrs"]  # (C, F, 3, 3)
    extrs = contents["extrs"]  # (C, F, 3, 4)
    camera_ids = [str(x) for x in contents["camera_ids"]]
    timestamps = contents.get("timestamps")
    episode = str(contents.get("episode", ""))

    cam_indices = _resolve_camera_indices(cameras, camera_ids)
    frame_indices = _resolve_frame_indices(frames, rgbs.shape[1])

    raw_views: List[Dict[str, Any]] = []
    for cam_idx in cam_indices:
        for frame_idx in frame_indices:
            rgb = rgbs[cam_idx, frame_idx].transpose(1, 2, 0)  # (H, W, 3)
            depth = depths[cam_idx, frame_idx, 0]  # (H, W)
            intrinsics = intrs[cam_idx, frame_idx]
            pose = _pose_3x4_to_4x4(extrs[cam_idx, frame_idx])

            view: Dict[str, Any] = {
                "img": rgb,
                "depth_z": depth,
                "intrinsics": intrinsics,
                "camera_poses": pose,
                "camera_id": camera_ids[cam_idx],
                "frame_index": frame_idx,
            }

            if timestamps is not None:
                view["timestamp"] = int(timestamps[frame_idx])

            raw_views.append(view)

    processed_views = preprocess_inputs(
        raw_views,
        resize_mode=resize_mode,
        size=size,
        norm_type=norm_type,
    )

    metadata = {
        "npz_path": str(npz_path),
        "episode": episode,
        "camera_ids": [camera_ids[idx] for idx in cam_indices],
        "frame_indices": frame_indices,
        "norm_type": norm_type,
        "resize_mode": resize_mode,
        "size": size,
    }

    return processed_views, metadata


__all__ = [
    "summarize_packed_npz",
    "views_from_packed_npz",
]

