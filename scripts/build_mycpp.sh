#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${FP_PYTHON:-python}"
cmake -S "${ROOT_DIR}/upstream/mycpp" -B "${ROOT_DIR}/upstream/mycpp/build" \
  -DPYTHON_EXECUTABLE="$(command -v "${PYTHON_BIN}")"
cmake --build "${ROOT_DIR}/upstream/mycpp/build" --parallel "${FP_BUILD_JOBS:-2}"
echo "mycpp built for ${PYTHON_BIN}"
