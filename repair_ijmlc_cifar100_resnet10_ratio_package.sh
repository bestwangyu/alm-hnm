#!/usr/bin/env bash
set -Eeuo pipefail

# Repackage an already completed experiment. This does not retrain anything.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

RUN_TAG="${RUN_TAG:-ijmlc_cifar100_resnet10_ratio_v1}"
GROUP="cifar100_resnet10"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="logs/${RUN_TAG}/${GROUP}"
SUMMARY="${LOG_ROOT}/${RUN_TAG}_${GROUP}_summary.csv"
PAIRED="${LOG_ROOT}/${RUN_TAG}_${GROUP}_paired.csv"
ARCHIVE="${RUN_TAG}_${GROUP}_results_no_pth_repacked.tar.gz"
CHECKSUM="${ARCHIVE}.sha256"

echo "[INFO] Repackaging existing results only"
echo "[INFO] repo=${REPO_DIR}"
echo "[INFO] log_root=${LOG_ROOT}"

[[ -d "${LOG_ROOT}" ]] || { echo "[ERROR] Missing ${LOG_ROOT}"; exit 2; }
[[ -f "${SUMMARY}" ]] || { echo "[ERROR] Missing ${SUMMARY}"; exit 2; }
[[ -f "${PAIRED}" ]] || { echo "[ERROR] Missing ${PAIRED}"; exit 2; }

"${PYTHON_BIN}" - "${SUMMARY}" "${PAIRED}" <<'PY'
import csv
import sys

summary, paired = sys.argv[1:]
with open(summary, newline="") as handle:
    rows = list(csv.DictReader(handle))
with open(paired, newline="") as handle:
    paired_rows = list(csv.DictReader(handle))

if len(rows) != 4:
    raise SystemExit(f"Expected 4 summary rows, got {len(rows)}")
if len(paired_rows) != 10:
    raise SystemExit(f"Expected 10 paired rows, got {len(paired_rows)}")
expected = {(ratio, method) for ratio in ("20", "100") for method in ("all", "semi_hard_topk")}
observed = {(row["ratio"], row["method"]) for row in rows}
if observed != expected:
    raise SystemExit(f"Unexpected conditions: {sorted(observed)}")
print("Result validation passed: 4 summary rows and 10 paired rows")
PY

checkpoint_paths=()
for ratio in 20 100; do
  for method in all semi_hard_topk; do
    path="checkpoint/${RUN_TAG}_${GROUP}_r${ratio}_${method}_"
    [[ -d "${path}" ]] || { echo "[ERROR] Missing checkpoint directory: ${path}"; exit 2; }
    checkpoint_paths+=("${path}")
  done
done

find "${LOG_ROOT}" "${checkpoint_paths[@]}" -type f ! -name '*.pth' -printf '%p\t%s bytes\n' \
  | sort > "${LOG_ROOT}/RESULT_MANIFEST_repacked.txt"
tar --exclude='*.pth' -czf "${ARCHIVE}" "${LOG_ROOT}" "${checkpoint_paths[@]}"
sha256sum "${ARCHIVE}" > "${CHECKSUM}"

echo "[INFO] Repackaging completed"
echo "[INFO] Archive: ${ARCHIVE}"
echo "[INFO] Checksum: ${CHECKSUM}"
