#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
PROFILE="${PROFILE:-ablation}"
MODEL_GROUP="${MODEL_GROUP:-ablation}"
OUTPUT_DIR="${OUTPUT_DIR:-code/results}"
DATA_ROOT="${DATA_ROOT:-dataset}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif [[ "${CONDA_DEFAULT_ENV:-}" == "kaf_profiti" ]]; then
  PYTHON_CMD=("python")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=("conda" "run" "-n" "kaf_profiti" "python")
elif [[ -x "/opt/anaconda3/envs/kaf_profiti/bin/python" ]]; then
  PYTHON_CMD=("/opt/anaconda3/envs/kaf_profiti/bin/python")
else
  echo "未找到 kaf_profiti conda 环境。请先运行: conda env create -f environment.yml" >&2
  exit 1
fi

PYTHONPATH=code "${PYTHON_CMD[@]}" code/run_experiment_matrix.py \
  --profile "$PROFILE" \
  --model-group "$MODEL_GROUP" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "$@"
