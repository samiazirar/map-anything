#!/usr/bin/env python3
"""
Utility for unpacking packed NPZ bundles (RGB, depth, intrinsics, extrinsics)
into MapAnything-ready view dictionaries.

Example usage:
    python scripts/inference_with_npz.py --npz /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/packed_npz/task_0065_user_0010_scene_0009_cfg_0004.npz --raw

    python scripts/inference_with_npz.py --npz /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/packed_npz/task_0065_user_0010_scene_0009_cfg_0004.npz --raw

    python scripts/inference_with_npz.py --npz /data/rh20t_api/test_npz/task_0065_user_0010_scene_0009_cfg_0004_processed.npz --raw  --depth-vis-dir data/depth_vis --save-pred-npz data/npz_files/task_0065_user_0010_scene_0009_cfg_0004_processed.npz

    The emitted ``views`` list can be passed directly to ``MapAnything.infer``.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from mapanything.utils import packed_npz as packed_npz_module
from mapanything.utils.image import preprocess_inputs

raw_views_from_packed_npz = packed_npz_module.raw_views_from_packed_npz
summarize_packed_npz = packed_npz_module.summarize_packed_npz
views_from_packed_npz = packed_npz_module.views_from_packed_npz

if hasattr(packed_npz_module, "load_packed_npz"):
    load_packed_npz = packed_npz_module.load_packed_npz
else:

    def load_packed_npz(npz_path: Path) -> Dict[str, Any]:
        with np.load(npz_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}


# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import torch.nn.functional as F
from mapanything.models import MapAnything

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert packed RGB/depth NPZ bundles into MapAnything view dictionaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              # Basic inspection of a packed NPZ bundle
              python scripts/inference_with_npz.py --npz /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/packed_npz/task_0065_user_0010_scene_0009_cfg_0004.npz --raw

              # Also dump depth visualizations and save a predictions NPZ
              python scripts/inference_with_npz.py --npz /data/rh20t_api/data/test_data_full_rgb_upscaled_depth/packed_npz/task_0065_user_0010_scene_0009_cfg_0004.npz --raw  --depth-vis-dir data/depth_vis --save-pred-npz data/npz_files/task_0065_user_0010_scene_0009_cfg_0004_pred.npz
            """
        ),
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
        default="keep",
        choices=["keep", "fixed_mapping", "longest_side", "square", "fixed_size"],
        help="Resize behaviour passed to preprocess_inputs (use 'keep' to preserve native resolution)",
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
    parser.add_argument(
        "--depth-vis-dir",
        type=Path,
        default=None,
        help="Optional directory to save depth visualizations (input vs prediction)",
    )
    parser.add_argument(
        "--save-pred-npz",
        type=Path,
        default=None,
        help="Optional output path for an NPZ containing the original data plus predicted depths",
    )

    return parser.parse_args()


def _coerce_size_argument(resize_mode: str, size_arg: Optional[List[int]]) -> Optional[Any]:
    if resize_mode == "keep":
        if size_arg:
            raise ValueError("--size cannot be used when resize_mode=keep")
        return None

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


def _extract_depth_as_numpy(depth: Any) -> Optional[np.ndarray]:
    if depth is None:
        return None

    if torch.is_tensor(depth):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth)

    depth_np = np.squeeze(depth_np)
    if depth_np.ndim != 2:
        return None
    return depth_np.astype(np.float32, copy=False)


def _save_depth_visualization(depth: Any, output_path: Path) -> bool:
    depth_np = _extract_depth_as_numpy(depth)
    if depth_np is None:
        return False

    valid_mask = np.isfinite(depth_np) & (depth_np > 0)
    scaled = np.zeros_like(depth_np, dtype=np.float32)
    if np.any(valid_mask):
        valid_values = depth_np[valid_mask]
        dmin = valid_values.min()
        dmax = valid_values.max()
        if dmax - dmin < 1e-6:
            scaled[valid_mask] = 1.0
        else:
            scaled[valid_mask] = (depth_np[valid_mask] - dmin) / (dmax - dmin)

    image_array = (scaled * 255.0).clip(0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(output_path)
    return True


def _save_predictions_npz(
    original_npz: Path,
    output_path: Path,
    predictions: List[Dict[str, torch.Tensor]],
    view_infos: List[Dict[str, Any]],
) -> None:
    if not predictions:
        raise ValueError("No predictions available to save")

    contents = load_packed_npz(original_npz)
    if "depths" not in contents:
        raise ValueError("Original NPZ does not contain 'depths'")

    camera_ids = [str(x) for x in contents.get("camera_ids", [])]
    camera_lookup = {cam_id: idx for idx, cam_id in enumerate(camera_ids)}

    pred_depths = contents["depths"].astype(np.float32, copy=True)

    for info, pred in zip(view_infos, predictions):
        depth_pred = pred.get("depth_z")
        if depth_pred is None:
            continue

        cam_id = info.get("camera_id")
        frame_idx = info.get("frame_index")
        original_size = info.get("original_size")
        if cam_id not in camera_lookup or frame_idx is None:
            continue

        cam_idx = camera_lookup[cam_id]
        target_shape = pred_depths[cam_idx, frame_idx, 0].shape

        depth_tensor = depth_pred
        if depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.unsqueeze(1)
        elif depth_tensor.ndim == 4 and depth_tensor.shape[-1] == 1:
            depth_tensor = depth_tensor.permute(0, 3, 1, 2)
        elif depth_tensor.ndim != 4:
            continue

        depth_tensor = depth_tensor.detach().to(dtype=torch.float32)
        if depth_tensor.is_cuda:
            depth_tensor = depth_tensor.cpu()

        if depth_tensor.shape[-2:] != target_shape:
            Warning(
                f"Resizing predicted depth from {depth_tensor.shape[-2:]} to {target_shape} "
                f"to match original size {original_size}"
            )
            depth_tensor = F.interpolate(
                depth_tensor,
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )

        depth_np = depth_tensor.squeeze(0).squeeze(0).numpy()
        pred_depths[cam_idx, frame_idx, 0] = depth_np

    contents["pred_depths"] = pred_depths

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **contents)


def main() -> None:
    args = parse_args()

    if args.raw:
        if args.size:
            print("[warning] --size is ignored when --raw is set.")
        if args.resize_mode != "keep":
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
    if not metadata["frame_indices"]:
        raise ValueError("No frame indices available for inference")

    frame0 = metadata["frame_indices"][0]
    test_views: List[Dict[str, Any]] = []
    view_infos: List[Dict[str, Any]] = []
    raw_depth_for_vis: Optional[np.ndarray] = None

    for original_view in views:
        if original_view["frame_index"] != frame0:
            continue

        view_copy = dict(original_view)
        camera_id = view_copy.get("camera_id")
        frame_index = view_copy.get("frame_index")
        original_depth_np = _extract_depth_as_numpy(view_copy.get("depth_z"))

        if raw_depth_for_vis is None and original_depth_np is not None:
            raw_depth_for_vis = original_depth_np

        view_infos.append(
            {
                "camera_id": camera_id,
                "frame_index": frame_index,
                "original_size": None if original_depth_np is None else original_depth_np.shape,
            }
        )

        view_copy.pop("frame_index", None)
        view_copy.pop("timestamp", None)
        view_copy.pop("camera_id", None)
        test_views.append(view_copy)

    if not test_views:
        raise ValueError("No views selected for inference after filtering")

    # Preprocess views for MapAnything when starting from raw NPZ content
    if args.raw:
        processed_views = preprocess_inputs(test_views,resize_mode="fixed_size",size=(630, 364),patch_size=14)     # 
    else:
        processed_views = test_views

    # Ensure depth maps broadcast correctly against ray directions
    for view in processed_views:
        depth = view.get("depth_z")
        if isinstance(depth, torch.Tensor) and depth.ndim == 3:
            view["depth_z"] = depth.unsqueeze(-1)
    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise Warning("Using CPU for inference. This is not recommended!")

    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
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
    if len(predictions) != len(view_infos):
        print(
            f"[warning] Expected {len(view_infos)} predictions but received {len(predictions)}; results will be truncated."
        )

    if args.depth_vis_dir is not None:
        before_path = args.depth_vis_dir / "depth_before.png"
        pred_path = args.depth_vis_dir / "depth_pred.png"

        saved_before = _save_depth_visualization(raw_depth_for_vis, before_path)

        pred_depth_for_vis: Optional[torch.Tensor] = None
        for pred in predictions:
            if "depth_z" in pred:
                pred_depth_for_vis = pred["depth_z"]
                break

        saved_pred = _save_depth_visualization(pred_depth_for_vis, pred_path)

        if saved_before or saved_pred:
            print(f"Saved depth visualizations to {args.depth_vis_dir}")
        else:
            print("No depth data available to visualize.")

    if args.save_pred_npz is not None:
        _save_predictions_npz(
            original_npz=args.npz,
            output_path=args.save_pred_npz,
            predictions=predictions,
            view_infos=view_infos,
        )
        print(f"Saved predictions NPZ to {args.save_pred_npz}")



if __name__ == "__main__":
    main()
