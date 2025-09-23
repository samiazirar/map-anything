"""Helpers for loading packed RGB/depth NPZ bundles.

The datasets bundled under ``packed_npz`` contain per-camera RGB frames,
depth maps, intrinsics, and extrinsics stored as numpy arrays.  This module
converts those bundles into the ``views`` structure expected by
``MapAnything.infer`` after running them through ``preprocess_inputs`` for
normalisation and resizing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from mapanything.utils.image import preprocess_inputs
import torch
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

def _load_npz(npz_path: Path) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_packed_npz(npz_path: Path) -> Dict[str, Any]:
    """Public helper to load packed NPZ bundles with pickle enabled."""

    return _load_npz(npz_path)


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


def _convert_views_without_resize(
    raw_views: List[Dict[str, Any]], norm_type: str
) -> List[Dict[str, Any]]:
    """Convert raw views to torch tensors without resizing or cropping."""

    if norm_type not in IMAGE_NORMALIZATION_DICT:
        raise ValueError(
            f"Unknown image normalization type: {norm_type}."
        )

    img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
    mean = img_norm.mean.view(1, -1, 1, 1)
    std = img_norm.std.view(1, -1, 1, 1)

    processed_views: List[Dict[str, Any]] = []

    for view in raw_views:
        base_view = dict(view)
        img = base_view.pop("img")

        if isinstance(img, torch.Tensor):
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"Expected image tensor with shape (H, W, 3); got {tuple(img.shape)}"
                )
            img_tensor = img.to(dtype=torch.float32)
        else:
            img_array = np.asarray(img)
            if img_array.ndim != 3 or img_array.shape[2] != 3:
                raise ValueError(
                    f"Expected image array with shape (H, W, 3); got {img_array.shape}"
                )
            img_tensor = torch.from_numpy(img_array).to(dtype=torch.float32)

        max_val = img_tensor.max()
        if max_val > 1.5:
            img_tensor = img_tensor / 255.0
        elif max_val > 1.0:
            img_tensor = img_tensor / max_val.clamp(min=1.0)

        img_tensor = img_tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
        normalized_img = (img_tensor - mean) / std

        processed_view: Dict[str, Any] = dict(base_view)
        processed_view["img"] = normalized_img
        processed_view["data_norm_type"] = [norm_type]
        processed_view["true_shape"] = np.int32([(img_tensor.shape[-2], img_tensor.shape[-1])])

        depth = processed_view.get("depth_z")
        if depth is not None:
            if torch.is_tensor(depth):
                depth_array = depth.detach().cpu().numpy().astype(np.float32)
            else:
                depth_array = np.asarray(depth, dtype=np.float32)
            if depth_array.ndim == 2:
                depth_array = depth_array[..., None]
            depth_tensor = torch.from_numpy(depth_array)
            if depth_tensor.ndim == 3:
                depth_tensor = depth_tensor.unsqueeze(0)
            processed_view["depth_z"] = depth_tensor.to(dtype=torch.float32)

        intrinsics = processed_view.get("intrinsics")
        if intrinsics is not None:
            if torch.is_tensor(intrinsics):
                intrinsics_tensor = intrinsics.detach().cpu().to(dtype=torch.float32)
            else:
                intrinsics_tensor = torch.from_numpy(
                    np.asarray(intrinsics, dtype=np.float32)
                )
            if intrinsics_tensor.ndim == 2:
                intrinsics_tensor = intrinsics_tensor.unsqueeze(0)
            processed_view["intrinsics"] = intrinsics_tensor

        camera_poses = processed_view.get("camera_poses")
        if camera_poses is not None:
            if torch.is_tensor(camera_poses):
                camera_pose_tensor = camera_poses.detach().cpu()
            else:
                camera_pose_tensor = torch.from_numpy(
                    np.asarray(camera_poses, dtype=np.float32)
                )
            if camera_pose_tensor.ndim == 2:
                camera_pose_tensor = camera_pose_tensor.unsqueeze(0)
            processed_view["camera_poses"] = camera_pose_tensor.to(dtype=torch.float32)

        is_metric_scale = processed_view.get("is_metric_scale")
        if is_metric_scale is None:
            processed_view["is_metric_scale"] = torch.ones(1, dtype=torch.bool)
        elif not torch.is_tensor(is_metric_scale):
            processed_view["is_metric_scale"] = torch.tensor(
                [bool(is_metric_scale)], dtype=torch.bool
            )
        elif is_metric_scale.ndim == 0:
            processed_view["is_metric_scale"] = is_metric_scale.view(1)

        processed_views.append(processed_view)

    return processed_views


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


def raw_views_from_packed_npz(
    npz_path: Path,
    cameras: Optional[Sequence[str]] = None,
    frames: Optional[Sequence[str]] = None,
    include_depth: bool = True,
    include_pose: bool = True,
    include_metric_scale: bool = True,
    group_by_timestamp: bool = False,
) -> Tuple[Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]], Dict[str, Any]]:
    """Return per-view dictionaries matching the README examples.

    If ``group_by_timestamp`` is True the result is a list of lists, where the
    outer index corresponds to the frame/timestamp and the inner list contains
    all camera views captured at that instant.
    """

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

    def iter_views() -> Iterable[Tuple[int, Dict[str, Any]]]:
        for frame_idx in frame_indices:
            for cam_idx in cam_indices:
                rgb = rgbs[cam_idx, frame_idx].transpose(1, 2, 0)  # (H, W, 3)
                intrinsics = intrs[cam_idx, frame_idx]
                view: Dict[str, Any] = {
                    "img": rgb,
                    "intrinsics": intrinsics,
                    "camera_id": camera_ids[cam_idx],
                    "frame_index": frame_idx,
                }

                if include_depth:
                    depth = depths[cam_idx, frame_idx, 0]
                    view["depth_z"] = depth
                    if include_metric_scale:
                        view["is_metric_scale"] = torch.tensor(True)

                if include_pose:
                    pose = _pose_3x4_to_4x4(extrs[cam_idx, frame_idx])
                    view["camera_poses"] = pose

                if timestamps is not None:
                    view["timestamp"] = int(timestamps[frame_idx])

                yield frame_idx, view

    if group_by_timestamp:
        grouped: List[List[Dict[str, Any]]] = []
        frame_to_group: Dict[int, List[Dict[str, Any]]] = {}
        for frame_idx, view in iter_views():
            if frame_idx not in frame_to_group:
                frame_to_group[frame_idx] = []
            frame_to_group[frame_idx].append(view)
        for frame_idx in frame_indices:
            grouped.append(frame_to_group.get(frame_idx, []))
        views: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]] = grouped
    else:
        views = [view for _, view in iter_views()]

    metadata = {
        "npz_path": str(npz_path),
        "episode": episode,
        "camera_ids": [camera_ids[idx] for idx in cam_indices],
        "frame_indices": frame_indices,
        "grouped_by_timestamp": group_by_timestamp,
        "include_depth": include_depth,
        "include_pose": include_pose,
        "include_metric_scale": include_metric_scale,
    }

    return views, metadata


def views_from_packed_npz(
    npz_path: Path,
    cameras: Optional[Sequence[str]] = None,
    frames: Optional[Sequence[str]] = None,
    resize_mode: str = "fixed_mapping",
    size: Optional[Any] = None,
    norm_type: str = "dinov2",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert a packed NPZ file into MapAnything view dictionaries.

    This wraps ``raw_views_from_packed_npz`` and feeds the result through
    ``preprocess_inputs`` so the images are resized and normalised the same way
    as ``load_images`` does in the main codebase.

    Args:
        npz_path: Path to the packed NPZ file.
        cameras: Optional subset of cameras to include (indices or camera ids).
        frames: Optional subset of frame indices to include.
        resize_mode: Resize behaviour passed to preprocess_inputs.
        size: Optional size parameter (single int for square/longest_side, two ints for fixed_size).
        norm_type: Image normalization key (see IMAGE_NORMALIZATION_DICT).
    Returns:
        A tuple of (processed views, metadata).

    """

    raw_views, metadata = raw_views_from_packed_npz(
        npz_path=npz_path,
        cameras=cameras,
        frames=frames,
        include_depth=True,
        include_pose=True,
        include_metric_scale=True,
        group_by_timestamp=False,
    )

    if resize_mode == "keep":
        processed_views = _convert_views_without_resize(
            raw_views=raw_views, norm_type=norm_type
        )
    else:
        processed_views = preprocess_inputs(
            raw_views,
            resize_mode=resize_mode,
            size=size,
            norm_type=norm_type,
        )

        # Ensure depth maps broadcast with ray directions inside MapAnything
        for view in processed_views:
            depth = view.get("depth_z")
            if isinstance(depth, torch.Tensor) and depth.ndim == 3:
                view["depth_z"] = depth.unsqueeze(-1)

    metadata.update(
        {
            "norm_type": norm_type,
            "resize_mode": resize_mode,
            "size": size,
        }
    )

    return processed_views, metadata


__all__ = [
    "summarize_packed_npz",
    "raw_views_from_packed_npz",
    "load_packed_npz",
    "views_from_packed_npz",
]
