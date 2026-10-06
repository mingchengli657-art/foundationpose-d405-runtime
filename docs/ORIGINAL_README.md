# Foundation Pose portable runtime

This directory is the online 6D-pose layer between `object_modeling` and a
robot controller. It contains the FoundationPose source/weights in `upstream/`
and the ROS 2 / IPC adapter in `app/`. It does not contain robot-control code.

The interface between projects is a portable model package directory produced
by `object_modeling`. The package contains `model.yaml`, a relative `mesh`
entry, a centered PLY for inspection, and `object_foundationpose.obj` for
FoundationPose. The runtime never relies on a hard-coded object number or a
current working directory.

For objects with an unobservable rotational degree of freedom, a package may
also include `pose_stabilization` in `model.yaml`.  For example,
`model_005_final` declares an axial lock about model `+Z`, which keeps a
near-cylindrical bottle's displayed and published box from spinning while
leaving its translation, long-axis direction, and depth validation unchanged.
Other model packages remain unchanged unless they opt into this metadata.

## Model contract

From this directory, validate any model package first:

```bash
./scripts/validate_model.sh ../object_modeling/datasets/object_003/model_fixed
```

The validator checks meter units, triangle faces, watertightness, consistent
face winding, positive volume, and outward normals. A model with mixed normals
is rejected before CUDA initialization.

## Runtime terminals

Terminal 1 is the ROS 2 D405 I/O process. It publishes the camera-frame pose
on `/foundationpose/object_pose_camera` and a machine-readable JSON copy on
`/foundationpose/object_pose_json`:

```bash
unset ROS_DISTRO
source /opt/ros/humble/setup.bash
source "$HOME/.config/robot_competition/dependencies.sh"
cd "/path/to/Foundation Pose"
./scripts/start_ros_io.sh
```

Terminal 2 is the GPU FoundationPose worker. Keep the model path as a sibling
relative path when the three project directories are copied together:

```bash
cd "/path/to/Foundation Pose"
export FP_PYTHON=python                 # active foundationpose environment
export CUDA_HOME=/usr/local/cuda-12.8   # adjust on another computer
./scripts/start_worker.sh \
  ../object_modeling/datasets/object_003/model_fixed
```

The first worker frame opens a polygon mask window. Draw a tight mask, press
Enter, and then keep the camera/object steady while initial registration runs.
The worker then tracks subsequent frames. It writes the full homogeneous
`camera_T_object` matrix to `runtime/pose.json`; the ROS I/O process converts it
to `PoseStamped` and JSON without changing axes.

The live display uses a model-axis-aligned box, so its edges share the same
object frame as the colored axes and the published pose. The ROS bridge
defaults to a low-latency 30 Hz input with a two-frame FIFO while retaining
adaptive throttling. Override these values with `FP_CAPTURE_HZ` and
`FP_QUEUE_SIZE`. The live window updates every processed frame; the optional
`latest_overlay.png` snapshot is throttled to 2 Hz and can be changed with
`FP_OVERLAY_SAVE_HZ` (set it to 0 to disable snapshots).

The normal ROS output is:

```text
/foundationpose/object_pose_camera  geometry_msgs/msg/PoseStamped
/foundationpose/object_pose_json    std_msgs/msg/String
```

The first message uses `d405_color_optical_frame` and contains a
`T_camera_object` pose. A robot-side adapter must transform it with the
calibrated hand-eye matrix before commanding Piper:

```text
T_base_object = T_base_tcp * T_tcp_camera * T_camera_object
```

This runtime never enables Piper, publishes `/control/move_*`, or moves the
gripper. Those actions belong in `Piper_Control` and remain explicitly gated.

## Portability

`upstream/` includes the two official local weight directories and the source
tree needed by this project. The bundled `mycpp` shared object is for the
current x86_64/Python-3.11 environment; run `./scripts/build_mycpp.sh` on a
different ABI or architecture. The Conda environment itself is intentionally
not copied into this project.
