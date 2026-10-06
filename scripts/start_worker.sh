#!/usr/bin/env bash
# Launch the GPU worker from this directory. The model is supplied as a
# portable package directory and is resolved relative to its own metadata.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
UPSTREAM_DIR="${FP_UPSTREAM_DIR:-${ROOT_DIR}/upstream}"
CONDA_ENV="${FP_CONDA_ENV:-foundationpose}"
CONDA_ROOT="${FP_MINIFORGE_ROOT:-${MINIFORGE_ROOT:-}}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
set -u

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR=/tmp/foundationpose-matplotlib
if [[ -n "${FP_TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST="${FP_TORCH_CUDA_ARCH_LIST}"
fi
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${FP_MODEL:-}" ]]; then
  MODEL_PATH="${FP_MODEL}"
else
  MODEL_PATH="${1:-}"
  shift || true
fi
if [[ -z "${MODEL_PATH}" ]]; then
  echo "usage: $0 MODEL_PACKAGE [worker options...]" >&2
  echo "example: $0 /path/to/object_modeling/datasets/object_003/model" >&2
  exit 2
fi
if ! MODEL_PATH="$(realpath -e -- "${MODEL_PATH}")"; then
  echo "model package not found: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -d "${UPSTREAM_DIR}" ]]; then
  echo "FoundationPose source directory not found: ${UPSTREAM_DIR}" >&2
  exit 2
fi

depth_inlier_threshold_m="${FP_DEPTH_INLIER_THRESHOLD_M:-0.025}"
min_depth_inlier_ratio="${FP_MIN_DEPTH_INLIER_RATIO:-0.45}"
max_depth_median_residual_m="${FP_MAX_DEPTH_MEDIAN_RESIDUAL_M:-0.035}"
min_depth_coverage="${FP_MIN_DEPTH_COVERAGE:-0.70}"
overlay_save_hz="${FP_OVERLAY_SAVE_HZ:-2}"

echo "Depth gate: residual<=${depth_inlier_threshold_m}m, inliers>=${min_depth_inlier_ratio}, median<=${max_depth_median_residual_m}m, coverage>=${min_depth_coverage}"
echo "Live overlay: every processed frame; PNG snapshot<=${overlay_save_hz}Hz"

IPC_DIR="${FP_IPC_DIR:-${ROOT_DIR}/runtime}"
PYTHON_BIN="${FP_PYTHON:-python}"

cd "${UPSTREAM_DIR}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/../app/foundationpose_d405_worker.py" \
  --foundationpose-dir "${UPSTREAM_DIR}" \
  --model "${MODEL_PATH}" \
  --ipc-dir "${IPC_DIR}" \
  --max-source-gap-s 0 \
  --depth-inlier-threshold-m "$depth_inlier_threshold_m" \
  --min-depth-inlier-ratio "$min_depth_inlier_ratio" \
  --max-depth-median-residual-m "$max_depth_median_residual_m" \
  --min-depth-coverage "$min_depth_coverage" \
  --overlay-save-hz "$overlay_save_hz" \
  "$@"
