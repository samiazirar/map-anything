#!/usr/bin/env python3
"""
Utility for unpacking packed NPZ bundles (RGB, depth, intrinsics, extrinsics)
into MapAnything-ready view dictionaries.

Example usage:
    python scripts/inference_with_npz.py --npz /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/packed_npz/task_0065_user_0010_scene_0009_cfg_0004.npz --raw

The emitted ``views`` list can be passed directly to ``MapAnything.infer``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional

import torch

from mapanything.utils.packed_npz import (
    raw_views_from_packed_npz,
    summarize_packed_npz,
    views_from_packed_npz,
)
from mapanything.utils.image import preprocess_inputs


# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import torch
from mapanything.models import MapAnything

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert packed RGB/depth NPZ bundles into MapAnything view dictionaries."
    )
    parser.add_argument("--npz", type=Path, required=True, help="Path to the packed NPZ file")
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="Optional subset of cameras to include (indices or camera ids)",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help="Optional subset of frame indices to include",
    )
    parser.add_argument(
        "--resize-mode",
        default="fixed_mapping",
        choices=["fixed_mapping", "longest_side", "square", "fixed_size"],
        help="Resize behaviour passed to preprocess_inputs",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs="*",
        default=None,
        help="Optional size parameter (single int for square/longest_side, two ints for fixed_size)",
    )
    parser.add_argument(
        "--norm-type",
        default="dinov2",
        help="Image normalization key (see IMAGE_NORMALIZATION_DICT)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Return raw README-style views without preprocessing",
    )
    parser.add_argument(
        "--group-by-timestamp",
        action="store_true",
        help="When --raw is set, group views by timestamp",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Exclude depth maps from raw views",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Exclude camera poses from raw views",
    )
    parser.add_argument(
        "--no-metric-scale",
        action="store_true",
        help="Exclude the is_metric_scale flag from raw views",
    )
    parser.add_argument(
        "--save-pt",
        type=Path,
        default=None,
        help="Optional path to torch.save the processed views + metadata",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a JSON summary of the NPZ contents before processing",
    )

    return parser.parse_args()


def _coerce_size_argument(resize_mode: str, size_arg: Optional[List[int]]) -> Optional[Any]:
    if not size_arg:
        return None

    if resize_mode in {"fixed_mapping"}:
        if len(size_arg) > 1:
            raise ValueError(
                "--size accepts at most one integer when resize_mode=fixed_mapping"
            )
        return size_arg[0] if size_arg else None

    if resize_mode in {"square", "longest_side"}:
        if len(size_arg) != 1:
            raise ValueError(
                f"resize_mode={resize_mode} expects a single integer --size value"
            )
        return size_arg[0]

    if resize_mode == "fixed_size":
        if len(size_arg) != 2:
            raise ValueError("fixed_size resize requires two integers: --size WIDTH HEIGHT")
        return tuple(size_arg)

    raise ValueError(f"Unsupported resize_mode: {resize_mode}")


def main() -> None:
    args = parse_args()

    if args.raw:
        if args.size:
            print("[warning] --size is ignored when --raw is set.")
        if args.resize_mode != "fixed_mapping":
            print("[warning] --resize-mode is ignored when --raw is set.")

    size = None if args.raw else _coerce_size_argument(args.resize_mode, args.size)

    if args.summary:
        summary = summarize_packed_npz(args.npz)
        print(json.dumps(summary, indent=2))

    if args.raw:
        views, metadata = raw_views_from_packed_npz(
            npz_path=args.npz,
            cameras=args.cameras,
            frames=args.frames,
            include_depth=not args.no_depth,
            include_pose=not args.no_pose,
            include_metric_scale=not args.no_metric_scale,
            group_by_timestamp=args.group_by_timestamp,
        )
    else:
        views, metadata = views_from_packed_npz(
            npz_path=args.npz,
            cameras=args.cameras,
            frames=args.frames,
            resize_mode=args.resize_mode,
            size=size,
            norm_type=args.norm_type,
        )

    if args.raw and args.group_by_timestamp:
        print(
            f"Loaded views for {len(metadata['frame_indices'])} timestamps "
            f"({sum(len(group) for group in views)} total camera views)"
        )
    else:
        print(f"Loaded {len(views)} views from {args.npz}")

    if args.save_pt:
        payload = {"views": views, "metadata": metadata}
        torch.save(payload, args.save_pt)
        print(f"Saved processed views to {args.save_pt}")
    
    frame0 = metadata["frame_indices"][0]
    test_views = [view for view in views if view["frame_index"] == frame0]
    #remove frame_index, timestamp and camera_id from each view
    for view in test_views:
        view.pop("frame_index", None)
        view.pop("timestamp", None)
        view.pop("camera_id", None)
        # Preprocess views for MapAnything
    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise Warning("Using CPU for inference. This is not recommended!")

    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    processed_views = preprocess_inputs(test_views)

    # Init model - This requries internet access or the huggingface hub cache to be pre-downloaded
    # For Apache 2.0 license model, use "facebook/map-anything-apache"
    
    breakpoint()
    predictions = model.infer(
    processed_views,                  # Any combination of input views
    memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
    use_amp=True,                     # Use mixed precision inference (recommended)
    amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
    apply_mask=True,                  # Apply masking to dense geometry outputs
    mask_edges=True,                  # Remove edge artifacts by using normals and depth
    apply_confidence_mask=False,      # Filter low-confidence regions
    confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
    # Control which inputs to use/ignore
    # By default, all inputs are used when provided
    ignore_calibration_inputs=False,
    ignore_depth_inputs=False,
    ignore_pose_inputs=False,
    ignore_depth_scale_inputs=False,
    ignore_pose_scale_inputs=False,
)

    breakpoint()


if __name__ == "__main__":
    main()
