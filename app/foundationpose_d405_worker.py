#!/usr/bin/env python3
"""Run FoundationPose on RGB-D snapshots supplied by foundationpose_ros_io.py.

On the first frame, left-click a tight polygon around the object and press
Enter.  FoundationPose then registers once and tracks newer frames.  Press R
to re-register with a new mask or Q/Esc to stop.  The worker writes a camera-
frame pose for the ROS I/O node; it never imports ROS or commands the robot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from foundationpose_depth_gate import evaluate_depth_consistency
from foundationpose_ipc_queue import (
    clear_frame_queue,
    dequeue_frame,
    frame_files,
)
from model_package import require_foundationpose_mesh, resolve_model
from pose_stabilization import lock_axial_rotation

def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(encoded.tobytes())
    os.replace(tmp, path)


def depth_in_meters(depth_raw: np.ndarray, scale: float, max_depth_m: float) -> np.ndarray:
    depth = np.asarray(depth_raw, dtype=np.float32) * float(scale)
    valid = np.isfinite(depth) & (depth >= 0.05) & (depth <= max_depth_m)
    depth[~valid] = 0.0
    return depth


def stamp_ns(metadata: dict) -> int:
    return int(metadata["stamp_sec"]) * 1_000_000_000 + int(metadata["stamp_nanosec"])


def pose_jump(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    translation_m = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    relative_rotation = current[:3, :3] @ previous[:3, :3].T
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    return translation_m, rotation_deg


def choose_polygon_mask(color_bgr: np.ndarray) -> np.ndarray:
    window = "FoundationPose initial mask"
    points: list[tuple[int, int]] = []

    def mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    while True:
        canvas = color_bgr.copy()
        if points:
            polygon = np.asarray(points, dtype=np.int32)
            cv2.polylines(canvas, [polygon], len(points) >= 3, (0, 255, 255), 2)
            for point in points:
                cv2.circle(canvas, point, 4, (0, 255, 255), -1)
        cv2.putText(
            canvas,
            "Left: add  Right: undo  Enter: accept  R: reset  Q: quit",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13, 32) and len(points) >= 3:
            break
        if key in (ord("r"), ord("R")):
            points.clear()
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt
    mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    cv2.destroyWindow(window)
    # Process the destroy request before synchronous CUDA registration starts.
    # Otherwise this window may look frozen during first-kernel compilation.
    cv2.waitKey(1)
    return mask


def load_or_choose_mask(mask_path: Path | None, ipc: Path, color_bgr: np.ndarray) -> np.ndarray:
    if mask_path is not None and mask_path.is_file():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != color_bgr.shape[:2]:
            raise RuntimeError(f"mask size does not match current image: {mask_path}")
        return (mask > 0).astype(np.uint8)
    mask = choose_polygon_mask(color_bgr)
    atomic_write_png(ipc / "initial_mask.png", mask * 255)
    return mask


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose-dir", default=str(root.parent / "upstream"))
    parser.add_argument(
        "--model",
        default=None,
        help="object_modeling model package directory (preferred) or an OBJ file",
    )
    parser.add_argument(
        "--mesh",
        default=None,
        help="legacy direct OBJ path; use --model for portable model metadata",
    )
    parser.add_argument("--ipc-dir", default="/tmp/foundationpose_d405")
    parser.add_argument("--mask-file", default=None)
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--axis-length-m", type=float, default=0.08)
    parser.add_argument(
        "--lock-axial-rotation",
        action="store_true",
        help=(
            "Canonicalize an axially symmetric model around its declared model "
            "axis. A model package can enable this per object."
        ),
    )
    parser.add_argument(
        "--axial-symmetry-axis",
        type=float,
        nargs=3,
        default=None,
        metavar=("AXIS_X", "AXIS_Y", "AXIS_Z"),
        help="Object-frame axis used with --lock-axial-rotation (default: +Z).",
    )
    parser.add_argument(
        "--overlay-save-hz",
        type=float,
        default=2.0,
        help=(
            "Maximum rate for writing latest_overlay.png. The live OpenCV window "
            "still updates on every processed frame; 0 disables PNG snapshots."
        ),
    )
    parser.add_argument(
        "--max-source-gap-s",
        type=float,
        default=0.0,
        help=(
            "Force re-registration after a source-timestamp gap; 0 disables it. "
            "Keep disabled with the bounded FIFO because slow inference "
            "intentionally creates gaps when camera callbacks are skipped."
        ),
    )
    parser.add_argument("--max-translation-jump-m", type=float, default=0.10)
    parser.add_argument("--max-rotation-jump-deg", type=float, default=50.0)
    parser.add_argument(
        "--rejects-before-reregister",
        "--rejects-before-recovery",
        dest="rejects_before_reregister",
        type=int,
        default=2,
        help="Consecutive rejected poses before entering recovery hold.",
    )
    parser.add_argument(
        "--recovery-good-frames",
        type=int,
        default=3,
        help="Consecutive valid frames needed to leave recovery hold.",
    )
    parser.add_argument(
        "--auto-reregister-on-loss",
        action="store_true",
        help="Legacy interactive behavior: ask for a new mask after repeated rejects.",
    )
    parser.add_argument("--depth-inlier-threshold-m", type=float, default=0.025)
    parser.add_argument("--min-depth-inlier-ratio", type=float, default=0.45)
    parser.add_argument("--max-depth-median-residual-m", type=float, default=0.035)
    parser.add_argument("--min-depth-coverage", type=float, default=0.70)
    parser.add_argument("--depth-mask-erosion-px", type=int, default=2)
    parser.add_argument(
        "--no-depth-consistency",
        action="store_false",
        dest="depth_consistency",
        help="Disable rendered-model versus D405 depth validation.",
    )
    parser.set_defaults(depth_consistency=True)
    parser.add_argument("--debug", type=int, default=1)
    args = parser.parse_args()
    if args.overlay_save_hz < 0:
        parser.error("--overlay-save-hz must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    repo = Path(args.foundationpose_dir).expanduser().resolve()
    if args.model and args.mesh:
        raise ValueError("pass only one of --model and --mesh")
    if args.model:
        model_spec = resolve_model(args.model)
    elif args.mesh:
        model_spec = resolve_model(args.mesh)
    else:
        raise ValueError("a model package is required; pass --model MODEL_DIR")
    mesh_path = model_spec.mesh_path
    ipc = Path(args.ipc_dir).expanduser().resolve()
    mask_path = Path(args.mask_file).expanduser().resolve() if args.mask_file else None
    ipc.mkdir(parents=True, exist_ok=True)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this terminal; FoundationPose requires NVIDIA CUDA")

    stabilization = model_spec.metadata.get("pose_stabilization", {})
    if not isinstance(stabilization, dict):
        stabilization = {}
    stabilization_mode = str(stabilization.get("mode", "none")).lower()
    lock_axial = bool(
        args.lock_axial_rotation
        or args.axial_symmetry_axis is not None
        or stabilization_mode in {"axial", "axial_lock", "lock_axial_rotation"}
        or bool(stabilization.get("lock_axial_rotation", False))
    )
    axial_axis = np.asarray(
        args.axial_symmetry_axis
        if args.axial_symmetry_axis is not None
        else stabilization.get("axis", [0.0, 0.0, 1.0]),
        dtype=float,
    ).reshape(3)
    if lock_axial:
        axis_norm = float(np.linalg.norm(axial_axis))
        if not np.isfinite(axial_axis).all() or axis_norm < 1e-8:
            raise ValueError("configured axial symmetry axis must be finite and non-zero")
        axial_axis /= axis_norm

    convex_hull = model_spec.metadata.get("mesh_quality", {}).get("topology") == "convex_hull"
    mesh_quality = require_foundationpose_mesh(mesh_path, convex_hull=convex_hull)
    print(
        f"Model: {model_spec.model_id} frame={model_spec.model_frame} "
        f"mesh={mesh_path} extents_m={mesh_quality['extents_m']}"
    )
    if lock_axial:
        print(
            "Pose stabilization: axial rotation locked about model axis "
            f"{np.round(axial_axis, 6).tolist()} (translation and tilt preserved)"
        )

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from Utils import (
        draw_posed_3d_box,
        draw_xyz_axis,
        nvdiffrast_render,
        set_logging_format,
        set_seed,
    )
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor
    import nvdiffrast.torch as dr

    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError(f"FoundationPose needs a triangle mesh: {mesh_path}")
    # Portable object_modeling packages define meaningful model axes: for the
    # ChArUco-built models, +Z is normal to the board.  Draw the box directly
    # in that frame.  A minimum-volume OBB is misleading for nearly round
    # bottles because small hull asymmetries can tilt its eigenvectors even
    # when the mesh and the estimated pose are upright.
    bbox = np.asarray(mesh.bounds, dtype=np.float64).reshape(2, 3)
    print(
        "visualization box: model-axis aligned, "
        f"extents_m={np.round(bbox[1] - bbox[0], 6).tolist()}"
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"mesh: {mesh_path}")
    print("Loading scorer/refiner; the first initialization can take a while...")
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(ipc / "debug"),
        debug=args.debug,
        glctx=glctx,
    )
    print("FoundationPose ready; waiting for synchronized D405 frames...")

    (ipc / "worker_status.json").unlink(missing_ok=True)
    previous_sequence = -1
    previous_stamp_ns: int | None = None
    previous_source_callback: int | None = None
    registered = False
    last_accepted_pose: np.ndarray | None = None
    ema_track_ms: float | None = None
    consecutive_rejects = 0
    recovery_required = False
    recovery_good_frames = 0
    last_overlay_save_time = 0.0
    canonical_pose: np.ndarray | None = None
    try:
        while True:
            snapshot = dequeue_frame(ipc, previous_sequence)
            if snapshot is None:
                if cv2.waitKey(10) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                time.sleep(0.01)
                continue
            meta, color_bgr, depth_raw, _frame_path = snapshot
            sequence = int(meta["sequence"])
            source_callback = int(meta.get("source_callback", sequence))
            current_stamp_ns = stamp_ns(meta)
            source_gap_s = (
                None
                if previous_stamp_ns is None
                else max(0.0, (current_stamp_ns - previous_stamp_ns) * 1e-9)
            )
            sequence_gap = 1 if previous_sequence < 0 else sequence - previous_sequence
            source_frame_gap = (
                1
                if previous_source_callback is None
                else source_callback - previous_source_callback
            )

            if (
                registered
                and args.max_source_gap_s > 0.0
                and source_gap_s is not None
                and source_gap_s > args.max_source_gap_s
            ):
                print(
                    f"source gap {source_gap_s:.3f}s exceeds {args.max_source_gap_s:.3f}s; "
                    "forcing a safe re-registration"
                )
                registered = False
                mask_path = None
                estimator.pose_last = None
                last_accepted_pose = None
                canonical_pose = None
                consecutive_rejects = 0

            rgb = np.ascontiguousarray(color_bgr[..., ::-1])
            depth = depth_in_meters(depth_raw, float(meta["depth_scale"]), args.max_depth_m)
            K = np.asarray(meta["K"], dtype=np.float64).reshape(3, 3)

            mode = "register" if not registered else "track"
            internal_pose_before = (
                estimator.pose_last.clone()
                if mode == "track" and estimator.pose_last is not None
                else None
            )
            inference_started = time.perf_counter()
            if not registered:
                mask = load_or_choose_mask(mask_path, ipc, color_bgr)
                valid_depth_pixels = int(np.count_nonzero((mask > 0) & (depth > 0)))
                if valid_depth_pixels < 50:
                    raise RuntimeError(
                        f"only {valid_depth_pixels} valid depth pixels inside the initial mask"
                    )
                registering_view = color_bgr.copy()
                cv2.putText(
                    registering_view,
                    "REGISTERING - keep camera and object still...",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(
                    "FoundationPose D405 - pose only, no robot control",
                    registering_view,
                )
                cv2.waitKey(1)
                pose = estimator.register(
                    K=K,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask.astype(bool),
                    iteration=args.est_refine_iter,
                )
                registered = True
                mask_path = ipc / "initial_mask.png"
                print(f"registered on frame {sequence}; valid mask depth={valid_depth_pixels}")
                # Frames captured while the user drew the mask and the global
                # registration ran are stale.  Start the tracking FIFO afresh.
                stale = clear_frame_queue(ipc)
                if stale:
                    print(f"discarded {stale} stale frame(s) after registration")
            else:
                pose = estimator.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    iteration=args.track_refine_iter,
                )

            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            if mode == "track":
                ema_track_ms = (
                    inference_ms
                    if ema_track_ms is None
                    else 0.8 * ema_track_ms + 0.2 * inference_ms
                )

            raw_pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
            pose = raw_pose
            if lock_axial:
                if mode == "register" or canonical_pose is None:
                    # Keep this as a candidate until all depth/jump gates for
                    # the frame have passed. A rejected frame must not move
                    # the canonical reference used by the next frame.
                    candidate_pose = raw_pose.copy()
                else:
                    candidate_pose = lock_axial_rotation(
                        raw_pose, canonical_pose, axial_axis
                    )
                pose = candidate_pose
            accepted = True
            reject_reason = ""
            translation_jump_m = 0.0
            rotation_jump_deg = 0.0
            depth_result = None
            if args.depth_consistency:
                with torch.inference_mode():
                    _rendered_rgb, rendered_depth, _rendered_normal = nvdiffrast_render(
                        K=K,
                        H=depth.shape[0],
                        W=depth.shape[1],
                        ob_in_cams=estimator.pose_last.reshape(1, 4, 4),
                        glctx=glctx,
                        mesh_tensors=estimator.mesh_tensors,
                    )
                depth_result = evaluate_depth_consistency(
                    depth,
                    rendered_depth[0].detach().float().cpu().numpy(),
                    inlier_threshold_m=args.depth_inlier_threshold_m,
                    min_inlier_ratio=args.min_depth_inlier_ratio,
                    max_median_residual_m=args.max_depth_median_residual_m,
                    min_coverage=args.min_depth_coverage,
                    erosion_px=args.depth_mask_erosion_px,
                )
                if not depth_result.passed:
                    accepted = False
                    reject_reason = f"depth mismatch: {depth_result.reason}"
            if mode == "track" and last_accepted_pose is not None:
                translation_jump_m, rotation_jump_deg = pose_jump(last_accepted_pose, pose)
                if translation_jump_m > args.max_translation_jump_m:
                    accepted = False
                    part = (
                        f"translation jump {translation_jump_m:.3f}m > "
                        f"{args.max_translation_jump_m:.3f}m"
                    )
                    reject_reason = f"{reject_reason}; {part}" if reject_reason else part
                if rotation_jump_deg > args.max_rotation_jump_deg:
                    accepted = False
                    part = (
                        f"rotation jump {rotation_jump_deg:.1f}deg > "
                        f"{args.max_rotation_jump_deg:.1f}deg"
                    )
                    reject_reason = f"{reject_reason}; {part}" if reject_reason else part

            if accepted:
                if lock_axial:
                    canonical_pose = pose.copy()
                last_accepted_pose = pose.copy()
                consecutive_rejects = 0
                if recovery_required:
                    recovery_good_frames += 1
                    print(
                        f"recovery validation {recovery_good_frames}/"
                        f"{args.recovery_good_frames}"
                    )
                    if recovery_good_frames >= args.recovery_good_frames:
                        recovery_required = False
                        recovery_good_frames = 0
                        print("tracking recovered; pose publication resumed")
                else:
                    recovery_good_frames = 0
            else:
                consecutive_rejects += 1
                recovery_good_frames = 0
                if internal_pose_before is not None:
                    estimator.pose_last = internal_pose_before
                print(
                    f"REJECT frame {sequence}: {reject_reason}; "
                    f"count={consecutive_rejects}/{args.rejects_before_reregister}"
                )

            force_reregister = False
            if not accepted and consecutive_rejects >= args.rejects_before_reregister:
                if args.auto_reregister_on_loss:
                    force_reregister = True
                elif not recovery_required:
                    recovery_required = True
                    recovery_good_frames = 0
                    print(
                        "tracking lost; entering recovery hold. Return the robot to "
                        "the saved observation pose; no manual mask will be requested"
                    )

            publish_valid = bool(accepted and not recovery_required)

            vis_rgb = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=pose, bbox=bbox)
            vis_rgb = draw_xyz_axis(
                vis_rgb,
                # The box and axes share the exact centered mesh frame.
                ob_in_cam=pose,
                scale=args.axis_length_m,
                K=K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            vis_bgr = np.ascontiguousarray(vis_rgb[..., ::-1])
            # ``pose`` is used unchanged for both display and output.
            xyz = pose[:3, 3]
            cv2.putText(
                vis_bgr,
                f"camera xyz: {xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f} m",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            gap_text = "n/a" if source_gap_s is None else f"{source_gap_s * 1000.0:.0f}ms"
            cv2.putText(
                vis_bgr,
                f"{mode}={inference_ms:.0f}ms  source_dt={gap_text}  "
                f"FIFO={len(frame_files(ipc))}  cam_gap={source_frame_gap}",
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0) if accepted else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if depth_result is not None:
                depth_color = (0, 255, 0) if depth_result.passed else (0, 0, 255)
                cv2.putText(
                    vis_bgr,
                    f"depth med={depth_result.median_residual_m * 1000.0:.0f}mm  "
                    f"inliers={depth_result.inlier_ratio:.0%}  "
                    f"coverage={depth_result.coverage:.0%}",
                    (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    depth_color,
                    2,
                    cv2.LINE_AA,
                )
            if not accepted:
                cv2.rectangle(
                    vis_bgr,
                    (2, 2),
                    (vis_bgr.shape[1] - 3, vis_bgr.shape[0] - 3),
                    (0, 0, 255),
                    5,
                )
                cv2.putText(
                    vis_bgr,
                    "POSE REJECTED - not published",
                    (12, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif recovery_required:
                cv2.rectangle(
                    vis_bgr,
                    (2, 2),
                    (vis_bgr.shape[1] - 3, vis_bgr.shape[0] - 3),
                    (0, 165, 255),
                    5,
                )
                cv2.putText(
                    vis_bgr,
                    f"RECOVERY HOLD {recovery_good_frames}/{args.recovery_good_frames} "
                    "- return to observation pose",
                    (12, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            overlay_now = time.monotonic()
            if (
                args.overlay_save_hz > 0.0
                and overlay_now - last_overlay_save_time >= 1.0 / args.overlay_save_hz
            ):
                atomic_write_png(ipc / "latest_overlay.png", vis_bgr)
                last_overlay_save_time = overlay_now
            atomic_write_json(
                ipc / "pose.json",
                {
                    "valid": publish_valid,
                    "sequence": sequence,
                    "stamp_sec": int(meta["stamp_sec"]),
                    "stamp_nanosec": int(meta["stamp_nanosec"]),
                    "camera_frame": str(meta["camera_frame"]),
                    "object_frame": model_spec.model_frame,
                    "model_id": model_spec.model_id,
                    "model_mesh": os.path.relpath(mesh_path, model_spec.model_dir),
                    "mesh_quality": mesh_quality,
                    # ``pose`` is the camera_T_object transform for the
                    # centered mesh frame, optionally canonicalized under the
                    # model's declared axial symmetry. Display and output use
                    # this same transform without an extra box-frame rotation.
                    "camera_T_object": pose.tolist(),
                    "pose_stabilization": {
                        "mode": "axial_lock" if lock_axial else "none",
                        "axis": axial_axis.tolist() if lock_axial else None,
                    },
                    "mode": mode,
                    "inference_ms": inference_ms,
                    "reject_reason": reject_reason,
                    "recovery_required": recovery_required,
                    "recovery_good_frames": recovery_good_frames,
                    "depth_consistency": (
                        None if depth_result is None else depth_result.to_dict()
                    ),
                },
            )
            atomic_write_json(
                ipc / "worker_status.json",
                {
                    "sequence": sequence,
                    "mode": mode,
                    "accepted": bool(accepted),
                    "published": publish_valid,
                    "inference_ms": inference_ms,
                    "ema_track_ms": 0.0 if ema_track_ms is None else ema_track_ms,
                    "source_gap_s": source_gap_s,
                    "sequence_gap": sequence_gap,
                    "source_frame_gap": source_frame_gap,
                    "queue_remaining": len(frame_files(ipc)),
                    "translation_jump_m": translation_jump_m,
                    "rotation_jump_deg": rotation_jump_deg,
                    "consecutive_rejects": consecutive_rejects,
                    "recovery_required": recovery_required,
                    "recovery_good_frames": recovery_good_frames,
                    "depth_consistency": (
                        None if depth_result is None else depth_result.to_dict()
                    ),
                },
            )
            previous_sequence = sequence
            # Ignore the long pause spent drawing the mask/initializing CUDA;
            # the next newly captured frame starts a fresh tracking timeline.
            previous_stamp_ns = None if mode == "register" else current_stamp_ns
            previous_source_callback = None if mode == "register" else source_callback

            if force_reregister:
                print("too many rejected poses; draw a new mask on the next frame")
                registered = False
                mask_path = None
                estimator.pose_last = None
                last_accepted_pose = None
                canonical_pose = None
                consecutive_rejects = 0
                recovery_required = False
                recovery_good_frames = 0
                clear_frame_queue(ipc)

            cv2.imshow("FoundationPose D405 - pose only, no robot control", vis_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                registered = False
                mask_path = None
                estimator.pose_last = None
                last_accepted_pose = None
                canonical_pose = None
                consecutive_rejects = 0
                recovery_required = False
                recovery_good_frames = 0
                previous_stamp_ns = None
                previous_source_callback = None
                clear_frame_queue(ipc)
                print("re-registration requested")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("FoundationPose worker stopped; no robot command was sent")


if __name__ == "__main__":
    main()
