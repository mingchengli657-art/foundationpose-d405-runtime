#!/usr/bin/env python3
"""ROS 2 I/O bridge between the D405 and a Python 3.11 FoundationPose worker.

ROS Humble uses system Python 3.10 while the FoundationPose environment uses
Python 3.11.  This node keeps those ABIs separate with a small atomic-file IPC
directory.  It never sends any command to the robot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, String

from foundationpose_ipc_queue import (
    clear_frame_queue,
    enqueue_frame,
    frame_files,
)

class FoundationPoseRosIo(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("foundationpose_d405_io")
        self.args = args
        self.bridge = CvBridge()
        self.ipc = Path(args.ipc_dir).expanduser().resolve()
        self.ipc.mkdir(parents=True, exist_ok=True)
        # Never publish a pose left behind by an earlier worker session.
        (self.ipc / "pose.json").unlink(missing_ok=True)
        (self.ipc / "worker_status.json").unlink(missing_ok=True)
        for legacy_name in ("frame.json", "color_bgr.npy", "depth_raw.npy"):
            (self.ipc / legacy_name).unlink(missing_ok=True)
        clear_frame_queue(self.ipc)
        self.K: np.ndarray | None = None
        self.camera_frame = args.camera_frame
        self.sequence = 0
        self.source_callbacks = 0
        self.last_capture_time = 0.0
        self.last_log_time = 0.0
        self.last_published_sequence = -1
        self.last_recovery_state: bool | None = None
        self.queue_full_skips = 0
        self.effective_capture_hz = args.capture_hz
        self.last_rate_update = 0.0

        color_sub = Subscriber(
            self, Image, args.color_topic, qos_profile=qos_profile_sensor_data
        )
        depth_sub = Subscriber(
            self, Image, args.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=args.sync_slop_s
        )
        self.sync.registerCallback(self.rgbd_callback)
        self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.info_callback,
            qos_profile_sensor_data,
        )
        self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, 10)
        # PoseStamped is the normal ROS interface.  This parallel JSON topic
        # carries the model id and full homogeneous matrix for non-ROS robot
        # adapters, without putting a filesystem path into the contract.
        self.pose_json_pub = self.create_publisher(String, args.pose_json_topic, 10)
        self.recovery_pub = self.create_publisher(Bool, args.recovery_topic, 10)
        self.create_timer(0.05, self.publish_pose_if_ready)

        self.get_logger().info(f"color: {args.color_topic}")
        self.get_logger().info(f"aligned depth: {args.depth_topic}")
        self.get_logger().info(f"camera info: {args.camera_info_topic}")
        self.get_logger().info(f"recovery state: {args.recovery_topic}")
        self.get_logger().info(f"IPC: {self.ipc}")
        self.get_logger().info(
            f"continuous FIFO: max_hz={args.capture_hz:.1f}, queue={args.queue_size}, "
            f"adaptive_rate={args.adaptive_rate}"
        )
        self.get_logger().info("SAFETY: this node publishes pose only; it cannot command Piper")

    def info_callback(self, msg: CameraInfo) -> None:
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if msg.header.frame_id:
            self.camera_frame = msg.header.frame_id

    def rgbd_callback(self, color_msg: Image, depth_msg: Image) -> None:
        if self.K is None:
            return
        self.source_callbacks += 1
        now = time.monotonic()
        self.update_effective_capture_rate(now)
        if now - self.last_capture_time < 1.0 / self.effective_capture_hz:
            return
        queued = len(frame_files(self.ipc))
        if queued >= self.args.queue_size:
            self.queue_full_skips += 1
            if now - self.last_log_time > 2.0:
                self.get_logger().warning(
                    f"FIFO full ({queued}/{self.args.queue_size}); waiting for worker, "
                    f"camera callbacks skipped={self.queue_full_skips}"
                )
                self.last_log_time = now
            return
        try:
            color_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warning(f"image conversion failed: {exc}")
            return
        color_bgr = np.ascontiguousarray(color_bgr)
        depth_raw = np.ascontiguousarray(depth_raw)
        if depth_raw.ndim != 2 or color_bgr.shape[:2] != depth_raw.shape[:2]:
            self.get_logger().warning("color/depth mismatch; depth must be aligned to color")
            return

        depth_scale = 1.0 if np.issubdtype(depth_raw.dtype, np.floating) else self.args.depth_scale
        stamp = color_msg.header.stamp
        self.sequence += 1
        metadata = {
            "sequence": self.sequence,
            "source_callback": self.source_callbacks,
            "stamp_sec": int(stamp.sec),
            "stamp_nanosec": int(stamp.nanosec),
            "camera_frame": self.camera_frame,
            "height": int(color_bgr.shape[0]),
            "width": int(color_bgr.shape[1]),
            "K": self.K.tolist(),
            "depth_scale": float(depth_scale),
            "depth_dtype": str(depth_raw.dtype),
        }
        enqueue_frame(
            self.ipc,
            self.sequence,
            metadata,
            color_bgr,
            depth_raw,
        )
        self.last_capture_time = now
        if now - self.last_log_time > 2.0:
            queued = len(frame_files(self.ipc))
            self.get_logger().info(
                f"RGB-D frame {self.sequence}: {color_bgr.shape[1]}x{color_bgr.shape[0]}, "
                f"FIFO={queued}/{self.args.queue_size}, effective_hz={self.effective_capture_hz:.1f}, "
                f"frame={self.camera_frame}, depth={depth_raw.dtype}"
            )
            self.last_log_time = now

    def update_effective_capture_rate(self, now: float) -> None:
        if not self.args.adaptive_rate or now - self.last_rate_update < 1.0:
            return
        self.last_rate_update = now
        status_path = self.ipc / "worker_status.json"
        try:
            if time.time() - status_path.stat().st_mtime > 3.0:
                return
            status = json.loads(status_path.read_text(encoding="utf-8"))
            inference_ms = float(status["ema_track_ms"])
            if inference_ms <= 0:
                return
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
            return
        # Keep a little headroom so the FIFO absorbs jitter instead of growing.
        recommended = 0.85 * 1000.0 / inference_ms
        recommended = min(self.args.capture_hz, max(1.0, recommended))
        self.effective_capture_hz = 0.7 * self.effective_capture_hz + 0.3 * recommended

    def publish_pose_if_ready(self) -> None:
        path = self.ipc / "pose.json"
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            sequence = int(result["sequence"])
            recovery_required = bool(result.get("recovery_required", False))
            if recovery_required != self.last_recovery_state:
                recovery_msg = Bool()
                recovery_msg.data = recovery_required
                self.recovery_pub.publish(recovery_msg)
                self.last_recovery_state = recovery_required
            if sequence <= self.last_published_sequence or not result.get("valid", False):
                return
            T = np.asarray(result["camera_T_object"], dtype=np.float64).reshape(4, 4)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return

        msg = PoseStamped()
        msg.header.frame_id = str(result.get("camera_frame") or self.camera_frame)
        msg.header.stamp.sec = int(result.get("stamp_sec", 0))
        msg.header.stamp.nanosec = int(result.get("stamp_nanosec", 0))
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, T[:3, 3])
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = map(float, quat)
        self.pose_pub.publish(msg)
        json_msg = String()
        json_msg.data = json.dumps({
            "schema_version": 1,
            "model_id": str(result.get("model_id", "")),
            "model_frame": str(result.get("object_frame", "object_model_center")),
            "camera_frame": msg.header.frame_id,
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
            "camera_T_object": T.tolist(),
            "position_m": [float(x) for x in T[:3, 3]],
            "quaternion_xyzw": [float(x) for x in quat],
        }, ensure_ascii=False)
        self.pose_json_pub.publish(json_msg)
        self.last_published_sequence = sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc-dir", default="/tmp/foundationpose_d405")
    parser.add_argument("--color-topic", default="/camera/d405/color/image_raw")
    parser.add_argument(
        "--depth-topic", default="/camera/d405/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--camera-info-topic", default="/camera/d405/color/camera_info")
    parser.add_argument("--pose-topic", default="/foundationpose/object_pose_camera")
    parser.add_argument("--pose-json-topic", default="/foundationpose/object_pose_json")
    parser.add_argument("--recovery-topic", default="/foundationpose/recovery_required")
    parser.add_argument("--camera-frame", default="d405_color_optical_frame")
    parser.add_argument("--capture-hz", type=float, default=30.0)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument(
        "--no-adaptive-rate",
        action="store_false",
        dest="adaptive_rate",
        help="Disable capture-rate adaptation based on measured tracking time.",
    )
    parser.set_defaults(adaptive_rate=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--sync-slop-s", type=float, default=0.08)
    args = parser.parse_args()
    if args.capture_hz <= 0:
        parser.error("--capture-hz must be positive")
    if args.queue_size < 2:
        parser.error("--queue-size must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = FoundationPoseRosIo(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
