#!/usr/bin/env bash
set -euo pipefail

# IJMLC Stage 1: controlled five-way mining ablation.
#
# Groups:
#   all             - original ALM behaviour
#   random          - matched-count random negative selection
#   topk            - hard negative top-k
#   semi_hard       - semi-hard candidates only
#   semi_hard_topk  - proposed mixed semi-hard + hard top-k selection
#
# Every group uses the same BCE baseline, ALM configuration, data ratio,
# training budget, and seed set. GPU 0 and GPU 1 split seeds 31-50.
#
# Usage from the repository root:
#   bash run_ijmlc_ablation.sh
#
# Optional overrides:
#   PYTHON_BIN=python GPU0_ID=0 GPU1_ID=1 \
#   BATCH_SIZE_PER_GPU=128 NUM_WORKERS=8 \
#   NUM_EPOCHS=2 \
#   SEEDS_GPU0=31,32,33 SEEDS_GPU1=34,35 ALL_SEEDS=31,32,33,34,35 \
#   ABLATION_MODES="random" RUN_TAG=ijmlc_ablation_preflight \
#   bash run_ijmlc_ablation.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"

SEEDS_GPU0="${SEEDS_GPU0:-31,32,33,34,35,36,37,38,39,40}"
SEEDS_GPU1="${SEEDS_GPU1:-41,42,43,44,45,46,47,48,49,50}"
ALL_SEEDS="${ALL_SEEDS:-31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50}"
FIRST_SEED="${ALL_SEEDS%%,*}"
LAST_SEED="${ALL_SEEDS##*,}"
SEED_RANGE_TAG="seeds${FIRST_SEED}-${LAST_SEED}"

DATASET_NAME="${DATASET_NAME:-cifar100}"
RATIO_POS_TRAIN="${RATIO_POS_TRAIN:-200}"
BASELINE_CONFIG="${BASELINE_CONFIG:-baseline_bce}"
ALM_CONFIG="${ALM_CONFIG:-ALM_13_beg}"
DELTA_VALUE="${DELTA_VALUE:-0.12}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"

RUN_TAG="${RUN_TAG:-ijmlc_ablation_v1}"
LOG_ROOT="${LOG_ROOT:-logs/${RUN_TAG}}"
CHECKPOINT_SUBFOLDER_PREFIX="${CHECKPOINT_SUBFOLDER_PREFIX:-${RUN_TAG}_}"
ABLATION_MODES="${ABLATION_MODES:-all random topk semi_hard semi_hard_topk}"
mkdir -p "${LOG_ROOT}"

if [[ "${GPU0_ID}" == "${GPU1_ID}" ]]; then
  echo "GPU0_ID and GPU1_ID must identify two different GPUs." >&2
  exit 2
fi

"${PYTHON_BIN}" - "${SEEDS_GPU0}" "${SEEDS_GPU1}" "${ALL_SEEDS}" <<'PY'
import sys

def parse(text):
    values = [int(value) for value in text.split(",") if value]
    if len(values) != len(set(values)):
        raise SystemExit("Duplicate seed in list: {}".format(text))
    return set(values)

gpu0 = parse(sys.argv[1])
gpu1 = parse(sys.argv[2])
all_seeds = parse(sys.argv[3])
if gpu0 & gpu1:
    raise SystemExit("GPU seed lists overlap: {}".format(sorted(gpu0 & gpu1)))
if gpu0 | gpu1 != all_seeds:
    raise SystemExit(
        "GPU seed union does not match ALL_SEEDS: missing={}, extra={}".format(
            sorted(all_seeds - (gpu0 | gpu1)),
            sorted((gpu0 | gpu1) - all_seeds),
        )
    )
PY

# Archive the exact implementation used by this run. This is intentionally
# separate from checkpoints so it can be copied with the result package.
mkdir -p "${LOG_ROOT}/source_snapshot"
mkdir -p "${LOG_ROOT}/source_snapshot/configs/baseline"
mkdir -p "${LOG_ROOT}/source_snapshot/configs/ALM"
cp main.py "${LOG_ROOT}/source_snapshot/main.py"
cp utils/custom_losses.py "${LOG_ROOT}/source_snapshot/custom_losses.py"
cp utils/train_test_CIFAR.py "${LOG_ROOT}/source_snapshot/train_test_CIFAR.py"
cp utils/final_ensembling_results.py "${LOG_ROOT}/source_snapshot/final_ensembling_results.py"
cp utils/CIFAR100_dataset.py "${LOG_ROOT}/source_snapshot/CIFAR100_dataset.py"
cp "configs/baseline/${BASELINE_CONFIG}.json" "${LOG_ROOT}/source_snapshot/configs/baseline/"
cp "configs/ALM/${ALM_CONFIG}.json" "${LOG_ROOT}/source_snapshot/configs/ALM/"
cp environment.yml "${LOG_ROOT}/source_snapshot/environment.yml"
cp "${BASH_SOURCE[0]}" "${LOG_ROOT}/source_snapshot/run_ijmlc_ablation.sh"
find "${LOG_ROOT}/source_snapshot" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${LOG_ROOT}/source_snapshot/SHA256SUMS"

{
  uname -a
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" -m pip freeze
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi
} > "${LOG_ROOT}/environment.txt" 2>&1

SEED_SUMMARY="${LOG_ROOT}/${RUN_TAG}_seed_metrics.csv"
ENSEMBLE_SUMMARY="${LOG_ROOT}/${RUN_TAG}_ensemble_summary.csv"
AGG_SUMMARY="${LOG_ROOT}/${RUN_TAG}_aggregate_summary.csv"
PAIRED_SUMMARY="${LOG_ROOT}/${RUN_TAG}_paired_comparisons.csv"

cat > "${SEED_SUMMARY}" <<'CSV'
mode,seed,validation_auc_max,test_roc_auc,test_auprc,test_fpr_at_tpr90,test_fpr_at_tpr95,test_fpr_at_tpr98,metrics_file,config_snapshot
CSV

cat > "${ENSEMBLE_SUMMARY}" <<'CSV'
mode,seed_count,ensemble_roc_auc,ensemble_auprc,ensemble_fpr_at_tpr90,ensemble_fpr_at_tpr95,ensemble_fpr_at_tpr98,ensemble_log,metrics_file,config_snapshot
CSV

run_group() {
  local mode="$1"
  local topk="$2"
  local warmup="$3"
  local random_k="$4"
  local upper="$5"

  local model_name="${RUN_TAG}_${mode}"
  local subfolder_name="${CHECKPOINT_SUBFOLDER_PREFIX}${mode}_"
  local log_dir="${LOG_ROOT}/${mode}"
  local checkpoint_root="checkpoint/${subfolder_name}/${model_name}"
  mkdir -p "${log_dir}"

  local common_args=(
    --dataset_name "${DATASET_NAME}"
    --name_file_bs "${BASELINE_CONFIG}"
    --name_file_alm "${ALM_CONFIG}"
    --model_name "${model_name}"
    --subfolder_name "${subfolder_name}"
    --ratio_pos_train "${RATIO_POS_TRAIN}"
    --delta "${DELTA_VALUE}"
    --bsize "${BATCH_SIZE_PER_GPU}"
    --num_workers "${NUM_WORKERS}"
    --num_epochs "${NUM_EPOCHS}"
    --hnm_mode "${mode}"
    --hnm_topk "${topk}"
    --hnm_warmup_epochs "${warmup}"
    --hnm_random_k "${random_k}"
    --hnm_semi_hard_upper "${upper}"
  )

  echo "============================================================"
  echo "Running IJMLC ablation group: ${mode}"
  echo "model=${model_name}"
  echo "mode=${mode}, topk=${topk}, warmup=${warmup}, random_k=${random_k}, upper=${upper}"
  echo "GPU ${GPU0_ID}: seeds ${SEEDS_GPU0}"
  echo "GPU ${GPU1_ID}: seeds ${SEEDS_GPU1}"

  set +e
  "${PYTHON_BIN}" main.py \
    --train 1 \
    --gpu_id "${GPU0_ID}" \
    --seed_list "${SEEDS_GPU0}" \
    "${common_args[@]}" \
    > "${log_dir}/train_gpu${GPU0_ID}.log" 2>&1 &
  local pid0=$!

  "${PYTHON_BIN}" main.py \
    --train 1 \
    --gpu_id "${GPU1_ID}" \
    --seed_list "${SEEDS_GPU1}" \
    "${common_args[@]}" \
    > "${log_dir}/train_gpu${GPU1_ID}.log" 2>&1 &
  local pid1=$!

  wait "${pid0}"
  local status0=$?
  wait "${pid1}"
  local status1=$?
  set -e

  if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
    echo "Training failed for mode=${mode}: GPU${GPU0_ID}=${status0}, GPU${GPU1_ID}=${status1}" >&2
    return 1
  fi

  "${PYTHON_BIN}" main.py \
    --train 0 \
    --gpu_id "${GPU0_ID}" \
    --seed_list "${ALL_SEEDS}" \
    "${common_args[@]}" \
    > "${log_dir}/ensemble.log" 2>&1

  local metrics_file="${checkpoint_root}/${model_name}_eval_seed_metrics_${SEED_RANGE_TAG}.csv"
  local ensemble_metrics_file="${checkpoint_root}/${model_name}_eval_ensemble_metrics_${SEED_RANGE_TAG}.csv"
  local config_snapshot="${checkpoint_root}/${model_name}_eval_config_snapshot_${SEED_RANGE_TAG}.json"
  if [[ ! -f "${metrics_file}" || ! -f "${ensemble_metrics_file}" || ! -f "${config_snapshot}" ]]; then
    echo "Missing evaluation assets for mode=${mode}: ${checkpoint_root}" >&2
    return 1
  fi

  tail -n +2 "${metrics_file}" | cut -d, -f1-7 | while IFS=, read -r seed validation_auc test_roc_auc test_auprc fpr90 fpr95 fpr98; do
    [[ -z "${seed}" ]] && continue
    echo "${mode},${seed},${validation_auc},${test_roc_auc},${test_auprc},${fpr90},${fpr95},${fpr98},${metrics_file},${config_snapshot}" >> "${SEED_SUMMARY}"
  done

  local ensemble_values
  ensemble_values="$(tail -n 1 "${ensemble_metrics_file}" | tr -d '\r')"
  echo "${mode},${ensemble_values},${log_dir}/ensemble.log,${ensemble_metrics_file},${config_snapshot}" >> "${ENSEMBLE_SUMMARY}"

  echo "Finished mode=${mode}"
  echo "Ensemble metrics=${ensemble_values}"
}

# Use top-k=10 for all selected-count comparisons. All mining variants use
# warmup=5, and mixed keeps random_k=0, so the ablation isolates the selection
# rule rather than a different activation schedule or random mixing policy.
for mode in ${ABLATION_MODES}; do
  case "${mode}" in
    all)
      run_group "all" 10 0 0 0.48
      ;;
    random)
      run_group "random" 10 5 0 0.48
      ;;
    topk)
      run_group "topk" 10 5 0 0.48
      ;;
    semi_hard)
      run_group "semi_hard" 10 5 0 0.48
      ;;
    semi_hard_topk)
      run_group "semi_hard_topk" 10 5 0 0.48
      ;;
    *)
      echo "Unsupported ABLATION_MODES entry: ${mode}" >&2
      exit 2
      ;;
  esac
done

"${PYTHON_BIN}" - "${SEED_SUMMARY}" "${AGG_SUMMARY}" "${PAIRED_SUMMARY}" <<'PY'
import csv
import math
import sys

import numpy as np
from scipy import stats

seed_file, aggregate_file, paired_file = sys.argv[1:]
metrics = [
    "validation_auc_max",
    "test_roc_auc",
    "test_auprc",
    "test_fpr_at_tpr90",
    "test_fpr_at_tpr95",
    "test_fpr_at_tpr98",
]

with open(seed_file, newline="") as handle:
    rows = list(csv.DictReader(handle))

groups = {}
for row in rows:
    groups.setdefault(row["mode"], []).append(row)

def mean(values):
    return sum(values) / len(values)

def sample_std(values):
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))

fieldnames = ["mode", "n"]
for metric in metrics:
    fieldnames.extend([metric + "_mean", metric + "_std"])

with open(aggregate_file, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for mode in sorted(groups):
        group = groups[mode]
        output = {"mode": mode, "n": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            output[metric + "_mean"] = mean(values)
            output[metric + "_std"] = sample_std(values)
        writer.writerow(output)

paired_fields = [
    "reference_mode",
    "comparator_mode",
    "metric",
    "n",
    "mean_paired_difference_reference_minus_comparator",
    "ci95_low",
    "ci95_high",
    "paired_t_pvalue",
    "wilcoxon_pvalue",
]
with open(paired_file, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=paired_fields)
    writer.writeheader()
    reference_mode = "semi_hard_topk"
    if reference_mode in groups:
        reference = {int(row["seed"]): row for row in groups[reference_mode]}
        rng = np.random.RandomState(20260831)
        for comparator_mode in sorted(mode for mode in groups if mode != reference_mode):
            comparator = {int(row["seed"]): row for row in groups[comparator_mode]}
            shared_seeds = sorted(set(reference) & set(comparator))
            for metric in metrics:
                differences = np.array(
                    [
                        float(reference[seed][metric])
                        - float(comparator[seed][metric])
                        for seed in shared_seeds
                    ],
                    dtype=float,
                )
                if differences.size:
                    bootstrap_indices = rng.randint(
                        0, differences.size, size=(10000, differences.size)
                    )
                    bootstrap_means = differences[bootstrap_indices].mean(axis=1)
                    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
                    t_pvalue = (
                        float(stats.ttest_1samp(differences, 0.0).pvalue)
                        if differences.size > 1
                        else float("nan")
                    )
                    try:
                        wilcoxon_pvalue = float(stats.wilcoxon(differences).pvalue)
                    except ValueError:
                        wilcoxon_pvalue = 1.0
                    mean_difference = float(differences.mean())
                else:
                    mean_difference = ci_low = ci_high = t_pvalue = wilcoxon_pvalue = float("nan")
                writer.writerow(
                    {
                        "reference_mode": reference_mode,
                        "comparator_mode": comparator_mode,
                        "metric": metric,
                        "n": differences.size,
                        "mean_paired_difference_reference_minus_comparator": mean_difference,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "paired_t_pvalue": t_pvalue,
                        "wilcoxon_pvalue": wilcoxon_pvalue,
                    }
                )
PY

"${PYTHON_BIN}" - "${SEED_SUMMARY}" "${ENSEMBLE_SUMMARY}" "${ABLATION_MODES}" "${ALL_SEEDS}" <<'PY'
import csv
import sys

seed_file, ensemble_file, modes_text, seeds_text = sys.argv[1:]
expected_modes = modes_text.split()
expected_seeds = {int(value) for value in seeds_text.split(",") if value}

with open(seed_file, newline="") as handle:
    seed_rows = list(csv.DictReader(handle))
with open(ensemble_file, newline="") as handle:
    ensemble_rows = list(csv.DictReader(handle))

for mode in expected_modes:
    actual_seeds = {int(row["seed"]) for row in seed_rows if row["mode"] == mode}
    if actual_seeds != expected_seeds:
        missing = sorted(expected_seeds - actual_seeds)
        extra = sorted(actual_seeds - expected_seeds)
        raise SystemExit(
            "Incomplete seed metrics for {}: missing={}, extra={}".format(
                mode, missing, extra
            )
            )
    mode_rows = [row for row in seed_rows if row["mode"] == mode]
    for row in mode_rows:
        for metric in (
            "validation_auc_max",
            "test_roc_auc",
            "test_auprc",
            "test_fpr_at_tpr90",
            "test_fpr_at_tpr95",
            "test_fpr_at_tpr98",
        ):
            try:
                float(row[metric])
            except ValueError:
                raise SystemExit(
                    "Non-numeric {} for mode={}, seed={}".format(
                        metric, mode, row["seed"]
                    )
                )

ensemble_modes = [row["mode"] for row in ensemble_rows]
if sorted(ensemble_modes) != sorted(expected_modes):
    raise SystemExit(
        "Incomplete ensemble summary: expected={}, actual={}".format(
            expected_modes, ensemble_modes
        )
        )
for row in ensemble_rows:
    for metric in (
        "seed_count",
        "ensemble_roc_auc",
        "ensemble_auprc",
        "ensemble_fpr_at_tpr90",
        "ensemble_fpr_at_tpr95",
        "ensemble_fpr_at_tpr98",
    ):
        try:
            float(row[metric])
        except ValueError:
            raise SystemExit(
                "Non-numeric {} in ensemble row for mode={}".format(
                    metric, row["mode"]
                )
            )
PY

echo "============================================================"
echo "IJMLC ablation finished."
echo "Per-seed summary: ${SEED_SUMMARY}"
echo "Ensemble summary: ${ENSEMBLE_SUMMARY}"
echo "Aggregate summary: ${AGG_SUMMARY}"
echo "Paired comparisons: ${PAIRED_SUMMARY}"

RESULT_ARCHIVE="${RESULT_ARCHIVE:-${RUN_TAG}_results_no_pth.tar.gz}"
checkpoint_paths=()
for mode in ${ABLATION_MODES}; do
  checkpoint_paths+=("checkpoint/${CHECKPOINT_SUBFOLDER_PREFIX}${mode}_")
done
find "${LOG_ROOT}" "${checkpoint_paths[@]}" -type f ! -name '*.pth' -printf '%p\t%s bytes\n' \
  | sort \
  > "${LOG_ROOT}/RESULT_MANIFEST.txt"
tar --exclude='*.pth' -czf "${RESULT_ARCHIVE}" "${LOG_ROOT}" "${checkpoint_paths[@]}"
sha256sum "${RESULT_ARCHIVE}" > "${RESULT_ARCHIVE}.sha256"
echo "Portable result archive without model weights: ${RESULT_ARCHIVE}"
echo "Archive checksum: ${RESULT_ARCHIVE}.sha256"
