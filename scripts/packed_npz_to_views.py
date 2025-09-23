#!/usr/bin/env python3
"""
Utility for unpacking packed NPZ bundles (RGB, depth, intrinsics, extrinsics)
into MapAnything-ready view dictionaries.

Example usage:
    python scripts/packed_npz_to_views.py \
        --npz /data/.../task_0065_user_0010_scene_0009_cfg_0004.npz \
        --save-pt unpacked_views.pt

The emitted ``views`` list can be passed directly to ``MapAnything.infer``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional

import torch

from mapanything.utils.packed_npz import summarize_packed_npz, views_from_packed_npz


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
    size = _coerce_size_argument(args.resize_mode, args.size)

    if args.summary:
        summary = summarize_packed_npz(args.npz)
        print(json.dumps(summary, indent=2))

    views, metadata = views_from_packed_npz(
        npz_path=args.npz,
        cameras=args.cameras,
        frames=args.frames,
        resize_mode=args.resize_mode,
        size=size,
        norm_type=args.norm_type,
    )

    print(f"Loaded {len(views)} views from {args.npz}")

    if args.save_pt:
        payload = {"views": views, "metadata": metadata}
        torch.save(payload, args.save_pt)
        print(f"Saved processed views to {args.save_pt}")


if __name__ == "__main__":
    main()
