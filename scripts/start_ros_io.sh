#!/usr/bin/env bash
# ROS setup scripts legitimately probe variables that may not exist yet, so
# enable nounset only after the environment has been sourced.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ROS_DISTRO_NAME="humble"
ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO_NAME}/setup.bash}"
if [[ -f "${ROS_SETUP}" ]]; then
  # The adapter is built for Humble.  Do not let a previously sourced Noetic
  # shell leak AMENT paths into this process.
  unset ROS_DISTRO
  # shellcheck disable=SC1090
  source "${ROS_SETUP}"
fi
DEPENDENCIES_SH="${FP_DEPENDENCIES_SH:-}"
if [[ -n "${DEPENDENCIES_SH}" && -f "${DEPENDENCIES_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${DEPENDENCIES_SH}"
fi
set -u

export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/foundationpose_ros_logs}"
IPC_DIR="${FP_IPC_DIR:-${ROOT_DIR}/runtime}"
PYTHON_BIN="${FP_ROS_PYTHON:-python3}"
mkdir -p "$ROS_LOG_DIR"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/../app/foundationpose_ros_io.py" \
  --ipc-dir "${IPC_DIR}" \
  --capture-hz "${FP_CAPTURE_HZ:-30}" \
  --queue-size "${FP_QUEUE_SIZE:-2}" \
  "$@"
