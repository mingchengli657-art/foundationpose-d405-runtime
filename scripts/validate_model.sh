#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL_PATH="${1:?用法: $0 MODEL_PACKAGE_OR_OBJ}"
exec "${FP_PYTHON:-python}" "${SCRIPT_DIR}/../app/validate_model.py" "${MODEL_PATH}"
