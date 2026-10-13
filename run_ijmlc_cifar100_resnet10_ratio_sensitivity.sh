#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot IJMLC supplementary run:
# CIFAR100 + ResNet10 at 1:20 and 1:100, Original ALM vs semi-hard top-k.
# The existing generalization runner performs the paired 20-seed training,
# ensemble evaluation, statistics, source snapshot, and no-PTH packaging.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
SINGLE_GPU="${SINGLE_GPU:-0}"
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"
DOWNLOAD="${DOWNLOAD:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
RUN_TAG="${RUN_TAG:-ijmlc_cifar100_resnet10_ratio_v1}"
GROUP="cifar100_resnet10"
RATIOS="20 100"

LOG_ROOT="logs/${RUN_TAG}/${GROUP}"
MASTER_LOG="logs/${RUN_TAG}_master.log"
STATUS_FILE="${LOG_ROOT}/run_status.tsv"
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

STAGE="initializing"
LABEL="-"

on_error() {
  local code=$?
  echo "[ERROR] stage=${STAGE} label=${LABEL} exit_code=${code} timestamp=$(date -Is)"
  echo "[ERROR] See ${MASTER_LOG} and ${LOG_ROOT}/*.log"
  exit "${code}"
}
trap on_error ERR

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

fail() {
  echo "[FATAL] $*" >&2
  exit 2
}

require_file() {
  [[ -f "$1" ]] || fail "Missing required file: $1"
}

STAGE="preflight"
printf 'stage\tstatus\ttimestamp\tdetail\n' > "${STATUS_FILE}"
printf 'preflight\tSTART\t%s\t%s\n' "$(date -Is)" "${REPO_DIR}" >> "${STATUS_FILE}"
log "Starting ${RUN_TAG}"
log "group=${GROUP} ratios=${RATIOS} dataset=cifar100 resnet=resnet10"
log "GPU0=${GPU0_ID} GPU1=${GPU1_ID} single_gpu=${SINGLE_GPU}"
log "batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} epochs=${NUM_EPOCHS} download=${DOWNLOAD}"

require_file "main.py"
require_file "run_ijmlc_generalization.sh"
require_file "utils/CIFAR100_dataset.py"
require_file "configs/baseline/baseline_bce.json"
require_file "configs/ALM/ALM_13_beg.json"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python executable not found: ${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import importlib
for name in ("torch", "torchvision", "numpy", "pandas", "scipy", "sklearn"):
    importlib.import_module(name)
print("Python dependency preflight: OK")
PY

if [[ "${SINGLE_GPU}" != "1" ]]; then
  "${PYTHON_BIN}" - "${GPU0_ID}" "${GPU1_ID}" <<'PY'
import sys
import torch
ids = [int(x) for x in sys.argv[1:]]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; use SINGLE_GPU=1 only for an intentional CPU/single-GPU test")
count = torch.cuda.device_count()
if max(ids) >= count:
    raise SystemExit(f"Requested GPU ids {ids}, but only {count} visible CUDA devices are available")
print(f"CUDA preflight: {count} visible device(s), requested {ids}")
PY
else
  "${PYTHON_BIN}" - "${SINGLE_GPU_ID}" <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
device = int(sys.argv[1])
if device >= torch.cuda.device_count():
    raise SystemExit(f"Requested GPU id {device}, but only {torch.cuda.device_count()} are visible")
print(f"Single-GPU preflight: using cuda:{device}")
PY
fi

if [[ "${DOWNLOAD}" != "1" && ! -d "datasets/cifar100" ]]; then
  fail "datasets/cifar100 is missing. Re-run with DOWNLOAD=1 or copy the dataset first."
fi

printf 'preflight\tDONE\t%s\tdependencies and GPU checks passed\n' "$(date -Is)" >> "${STATUS_FILE}"

STAGE="experiment"
LABEL="all conditions"
printf 'experiment\tSTART\t%s\tall and semi_hard_topk at ratios 20,100\n' "$(date -Is)" >> "${STATUS_FILE}"

export GROUP
export RATIOS
export DOWNLOAD
export BATCH_SIZE
export NUM_WORKERS
export NUM_EPOCHS
export GPU0_ID
export GPU1_ID
export SINGLE_GPU
export SINGLE_GPU_ID
export RUN_TAG
export PYTHON_BIN
export SEEDS_GPU0="${SEEDS_GPU0:-71,72,73,74,75,76,77,78,79,80}"
export SEEDS_GPU1="${SEEDS_GPU1:-81,82,83,84,85,86,87,88,89,90}"
export ALL_SEEDS="${ALL_SEEDS:-71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90}"

set +e
if bash "${REPO_DIR}/run_ijmlc_generalization.sh"; then
  RUN_STATUS=0
else
  RUN_STATUS=$?
fi
set -e

if [[ "${RUN_STATUS}" != "0" ]]; then
  echo "[ERROR] Generalization runner failed with exit code ${RUN_STATUS}"
  printf 'experiment\tFAILED\t%s\texit_code=%s\n' "$(date -Is)" "${RUN_STATUS}" >> "${STATUS_FILE}"
  for f in "${LOG_ROOT}"/*/train_gpu*.log "${LOG_ROOT}"/*/ensemble.log; do
    [[ -f "${f}" ]] || continue
    echo "----- tail ${f} -----"
    tail -n 100 "${f}" || true
  done
  exit "${RUN_STATUS}"
fi

printf 'experiment\tDONE\t%s\ttraining, evaluation, and base summary completed\n' "$(date -Is)" >> "${STATUS_FILE}"

STAGE="validation"
SUMMARY="${LOG_ROOT}/${RUN_TAG}_${GROUP}_summary.csv"
PAIRED="${LOG_ROOT}/${RUN_TAG}_${GROUP}_paired.csv"
ARCHIVE="${RUN_TAG}_${GROUP}_results_no_pth.tar.gz"
CHECKSUM="${ARCHIVE}.sha256"

require_file "${SUMMARY}"
require_file "${PAIRED}"
require_file "${ARCHIVE}"
require_file "${CHECKSUM}"

"${PYTHON_BIN}" - "${SUMMARY}" "${PAIRED}" <<'PY'
import csv
import os
import sys

summary, paired = sys.argv[1:]
with open(summary, newline="") as handle:
    summary_rows = list(csv.DictReader(handle))
with open(paired, newline="") as handle:
    paired_rows = list(csv.DictReader(handle))

if len(summary_rows) != 4:
    raise SystemExit(f"Expected 4 summary rows, got {len(summary_rows)}")
if len(paired_rows) != 10:
    raise SystemExit(f"Expected 10 paired rows, got {len(paired_rows)}")
expected = {(ratio, method) for ratio in ("20", "100") for method in ("all", "semi_hard_topk")}
observed = {(row["ratio"], row["method"]) for row in summary_rows}
if observed != expected:
    raise SystemExit(f"Summary condition mismatch: {sorted(observed)}")
print("Result validation: 4 summary rows, 10 paired rows, all conditions present")
PY

sha256sum -c "${CHECKSUM}"
printf 'validation\tDONE\t%s\tCSV row counts and archive checksum passed\n' "$(date -Is)" >> "${STATUS_FILE}"

STAGE="completed"
log "Experiment completed successfully; the base runner created and validated the archive"
log "Summary: ${SUMMARY}"
log "Paired statistics: ${PAIRED}"
log "Master log: ${MASTER_LOG}"
log "Archive: ${ARCHIVE}"
log "Checksum: ${CHECKSUM}"
