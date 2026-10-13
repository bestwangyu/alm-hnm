#!/usr/bin/env bash
set -euo pipefail

# IJMLC Stage 2A: parameter robustness and mixed-mining boundary analysis.
#
# This is intentionally a new-seed experiment. It does not repeat the Stage 1
# five-way ablation (seeds 31-50). The predefined grid is fixed before looking
# at the results and contains one all-pair control, a 3x3 top-k/upper grid,
# and two actual random-mixing points.
#
# Default seeds:
#   GPU 0: 51-60
#   GPU 1: 61-70
#
# Usage:
#   bash run_ijmlc_stage2_robustness.sh
#
# Optional preflight:
#   SEEDS_GPU0=51 SEEDS_GPU1=52 ALL_SEEDS=51,52 \
#   EXPERIMENT_SPECS='all|all|0|0|0|0' NUM_EPOCHS=2 \
#   RUN_TAG=ijmlc_stage2_preflight bash run_ijmlc_stage2_robustness.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"

SEEDS_GPU0="${SEEDS_GPU0:-51,52,53,54,55,56,57,58,59,60}"
SEEDS_GPU1="${SEEDS_GPU1:-61,62,63,64,65,66,67,68,69,70}"
ALL_SEEDS="${ALL_SEEDS:-51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70}"
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
ALM_Q_CLIP="${ALM_Q_CLIP:-10.0}"
ALM_LAMBDA_CLIP="${ALM_LAMBDA_CLIP:-1.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"

RUN_TAG="${RUN_TAG:-ijmlc_stage2_robustness_v2}"
LOG_ROOT="${LOG_ROOT:-logs/${RUN_TAG}}"
CHECKPOINT_SUBFOLDER_PREFIX="${CHECKPOINT_SUBFOLDER_PREFIX:-${RUN_TAG}_}"

# label|mode|topk|upper|random_k|warmup
EXPERIMENT_SPECS="${EXPERIMENT_SPECS:-\
all|all|0|0|0|0 \
semi_topk_k5_u030|semi_hard_topk|5|0.30|0|5 \
semi_topk_k5_u048|semi_hard_topk|5|0.48|0|5 \
semi_topk_k5_u060|semi_hard_topk|5|0.60|0|5 \
semi_topk_k10_u030|semi_hard_topk|10|0.30|0|5 \
semi_topk_k10_u048|semi_hard_topk|10|0.48|0|5 \
semi_topk_k10_u060|semi_hard_topk|10|0.60|0|5 \
semi_topk_k20_u030|semi_hard_topk|20|0.30|0|5 \
semi_topk_k20_u048|semi_hard_topk|20|0.48|0|5 \
semi_topk_k20_u060|semi_hard_topk|20|0.60|0|5 \
semi_topk_k10_u048_r1|semi_hard_topk|10|0.48|1|5 \
semi_topk_k10_u048_r2|semi_hard_topk|10|0.48|2|5}"

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
cp "${BASH_SOURCE[0]}" "${LOG_ROOT}/source_snapshot/run_ijmlc_stage2_robustness.sh"
find "${LOG_ROOT}/source_snapshot" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "${LOG_ROOT}/source_snapshot/SHA256SUMS"

{
  uname -a
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" -m pip freeze
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
} > "${LOG_ROOT}/environment.txt" 2>&1

SEED_SUMMARY="${LOG_ROOT}/${RUN_TAG}_seed_metrics.csv"
ENSEMBLE_SUMMARY="${LOG_ROOT}/${RUN_TAG}_ensemble_summary.csv"
AGG_SUMMARY="${LOG_ROOT}/${RUN_TAG}_aggregate_summary.csv"
PAIRED_SUMMARY="${LOG_ROOT}/${RUN_TAG}_vs_all_paired.csv"

cat > "${SEED_SUMMARY}" <<'CSV'
label,mode,topk,semi_hard_upper,random_k,warmup,seed,validation_auc_max,test_roc_auc,test_auprc,test_fpr_at_tpr90,test_fpr_at_tpr95,test_fpr_at_tpr98,metrics_file,config_snapshot
CSV

cat > "${ENSEMBLE_SUMMARY}" <<'CSV'
label,mode,topk,semi_hard_upper,random_k,warmup,seed_count,ensemble_roc_auc,ensemble_auprc,ensemble_fpr_at_tpr90,ensemble_fpr_at_tpr95,ensemble_fpr_at_tpr98,ensemble_log,metrics_file,config_snapshot
CSV

run_experiment() {
  local label="$1"
  local mode="$2"
  local topk="$3"
  local upper="$4"
  local random_k="$5"
  local warmup="$6"
  local model_name="${RUN_TAG}_${label}"
  local subfolder_name="${CHECKPOINT_SUBFOLDER_PREFIX}${label}_"
  local log_dir="${LOG_ROOT}/${label}"
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
    --alm_q_clip "${ALM_Q_CLIP}"
    --alm_lambda_clip "${ALM_LAMBDA_CLIP}"
    --grad_clip_norm "${GRAD_CLIP_NORM}"
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
  echo "Running IJMLC Stage 2 experiment: ${label}"
  echo "mode=${mode}, topk=${topk}, upper=${upper}, random_k=${random_k}, warmup=${warmup}"
  echo "GPU ${GPU0_ID}: seeds ${SEEDS_GPU0}"
  echo "GPU ${GPU1_ID}: seeds ${SEEDS_GPU1}"

  set +e
  "${PYTHON_BIN}" main.py --train 1 --gpu_id "${GPU0_ID}" \
    --seed_list "${SEEDS_GPU0}" "${common_args[@]}" \
    > "${log_dir}/train_gpu${GPU0_ID}.log" 2>&1 &
  local pid0=$!
  "${PYTHON_BIN}" main.py --train 1 --gpu_id "${GPU1_ID}" \
    --seed_list "${SEEDS_GPU1}" "${common_args[@]}" \
    > "${log_dir}/train_gpu${GPU1_ID}.log" 2>&1 &
  local pid1=$!
  wait "${pid0}"; local status0=$?
  wait "${pid1}"; local status1=$?
  set -e
  if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
    echo "Training failed for ${label}: GPU${GPU0_ID}=${status0}, GPU${GPU1_ID}=${status1}" >&2
    return 1
  fi

  "${PYTHON_BIN}" main.py --train 0 --gpu_id "${GPU0_ID}" \
    --seed_list "${ALL_SEEDS}" "${common_args[@]}" \
    > "${log_dir}/ensemble.log" 2>&1

  local metrics_file="${checkpoint_root}/${model_name}_eval_seed_metrics_${SEED_RANGE_TAG}.csv"
  local ensemble_metrics_file="${checkpoint_root}/${model_name}_eval_ensemble_metrics_${SEED_RANGE_TAG}.csv"
  local config_snapshot="${checkpoint_root}/${model_name}_eval_config_snapshot_${SEED_RANGE_TAG}.json"
  if [[ ! -f "${metrics_file}" || ! -f "${ensemble_metrics_file}" || ! -f "${config_snapshot}" ]]; then
    echo "Missing evaluation assets for ${label}: ${checkpoint_root}" >&2
    return 1
  fi

  tail -n +2 "${metrics_file}" | cut -d, -f1-7 | while IFS=, read -r seed validation_auc test_roc_auc test_auprc fpr90 fpr95 fpr98; do
    [[ -z "${seed}" ]] && continue
    echo "${label},${mode},${topk},${upper},${random_k},${warmup},${seed},${validation_auc},${test_roc_auc},${test_auprc},${fpr90},${fpr95},${fpr98},${metrics_file},${config_snapshot}" >> "${SEED_SUMMARY}"
  done

  local ensemble_values
  ensemble_values="$(tail -n 1 "${ensemble_metrics_file}" | tr -d '\r')"
  echo "${label},${mode},${topk},${upper},${random_k},${warmup},${ensemble_values},${log_dir}/ensemble.log,${ensemble_metrics_file},${config_snapshot}" >> "${ENSEMBLE_SUMMARY}"
  echo "Finished ${label}: ${ensemble_values}"
}

for spec in ${EXPERIMENT_SPECS}; do
  IFS='|' read -r label mode topk upper random_k warmup <<< "${spec}"
  run_experiment "${label}" "${mode}" "${topk}" "${upper}" "${random_k}" "${warmup}"
done

"${PYTHON_BIN}" - "${SEED_SUMMARY}" "${ENSEMBLE_SUMMARY}" "${AGG_SUMMARY}" "${PAIRED_SUMMARY}" "${ALL_SEEDS}" <<'PY'
import csv
import math
import sys

import numpy as np
from scipy import stats

seed_file, ensemble_file, aggregate_file, paired_file, seeds_text = sys.argv[1:]
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
with open(ensemble_file, newline="") as handle:
    ensemble_rows = list(csv.DictReader(handle))

groups = {}
for row in rows:
    groups.setdefault(row["label"], []).append(row)

def sample_std(values):
    if len(values) < 2:
        return 0.0
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))

aggregate_fields = ["label", "mode", "topk", "semi_hard_upper", "random_k", "warmup", "n"]
for metric in metrics:
    aggregate_fields.extend([metric + "_mean", metric + "_std"])

with open(aggregate_file, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
    writer.writeheader()
    for label in sorted(groups):
        group = groups[label]
        output = {key: group[0][key] for key in ["label", "mode", "topk", "semi_hard_upper", "random_k", "warmup"]}
        output["n"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            output[metric + "_mean"] = sum(values) / len(values)
            output[metric + "_std"] = sample_std(values)
        writer.writerow(output)

paired_fields = [
    "reference_label", "comparator_label", "metric", "n",
    "mean_paired_difference_reference_minus_comparator",
    "ci95_low", "ci95_high", "paired_t_pvalue", "wilcoxon_pvalue",
]
with open(paired_file, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=paired_fields)
    writer.writeheader()
    if "all" not in groups:
        raise SystemExit("The Stage 2 control label 'all' is missing.")
    reference = {int(row["seed"]): row for row in groups["all"]}
    rng = np.random.RandomState(20260901)
    for label in sorted(groups):
        if label == "all":
            continue
        comparator = {int(row["seed"]): row for row in groups[label]}
        shared = sorted(set(reference) & set(comparator))
        if not shared:
            raise SystemExit("No shared seeds for comparison with {}".format(label))
        for metric in metrics:
            differences = np.array(
                [float(reference[s][metric]) - float(comparator[s][metric]) for s in shared],
                dtype=float,
            )
            indices = rng.randint(0, len(differences), size=(10000, len(differences)))
            bootstrap = differences[indices].mean(axis=1)
            ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
            t_p = float(stats.ttest_1samp(differences, 0.0).pvalue) if len(differences) > 1 else float("nan")
            try:
                w_p = float(stats.wilcoxon(differences).pvalue)
            except ValueError:
                w_p = 1.0
            writer.writerow({
                "reference_label": "all",
                "comparator_label": label,
                "metric": metric,
                "n": len(differences),
                "mean_paired_difference_reference_minus_comparator": float(differences.mean()),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "paired_t_pvalue": t_p,
                "wilcoxon_pvalue": w_p,
            })

expected_seeds = {int(value) for value in seeds_text.split(",") if value}
for label, group in groups.items():
    actual = {int(row["seed"]) for row in group}
    if expected_seeds is not None and actual != expected_seeds:
        raise SystemExit("Incomplete seeds for {}: missing={}, extra={}".format(
            label, sorted(expected_seeds - actual), sorted(actual - expected_seeds)))
PY

"${PYTHON_BIN}" - "${SEED_SUMMARY}" "${ENSEMBLE_SUMMARY}" "${EXPERIMENT_SPECS}" "${ALL_SEEDS}" <<'PY'
import csv
import sys

seed_file, ensemble_file, specs_text, seeds_text = sys.argv[1:]
labels = [spec.split("|")[0] for spec in specs_text.split()]
expected = {int(value) for value in seeds_text.split(",") if value}
with open(seed_file, newline="") as handle:
    seeds = list(csv.DictReader(handle))
with open(ensemble_file, newline="") as handle:
    ensembles = list(csv.DictReader(handle))
for label in labels:
    actual = {int(row["seed"]) for row in seeds if row["label"] == label}
    if actual != expected:
        raise SystemExit("Incomplete seed summary for {}".format(label))
if sorted(row["label"] for row in ensembles) != sorted(labels):
    raise SystemExit("Incomplete ensemble summary")
PY

RESULT_ARCHIVE="${RESULT_ARCHIVE:-${RUN_TAG}_results_no_pth.tar.gz}"
checkpoint_paths=()
for spec in ${EXPERIMENT_SPECS}; do
  label="${spec%%|*}"
  checkpoint_paths+=("checkpoint/${CHECKPOINT_SUBFOLDER_PREFIX}${label}_")
done
find "${LOG_ROOT}" "${checkpoint_paths[@]}" -type f ! -name '*.pth' -printf '%p\t%s bytes\n' \
  | sort > "${LOG_ROOT}/RESULT_MANIFEST.txt"
tar --exclude='*.pth' -czf "${RESULT_ARCHIVE}" "${LOG_ROOT}" "${checkpoint_paths[@]}"
sha256sum "${RESULT_ARCHIVE}" > "${RESULT_ARCHIVE}.sha256"

echo "============================================================"
echo "IJMLC Stage 2 robustness experiment finished."
echo "Seed summary: ${SEED_SUMMARY}"
echo "Aggregate summary: ${AGG_SUMMARY}"
echo "Paired comparison: ${PAIRED_SUMMARY}"
echo "Portable result archive: ${RESULT_ARCHIVE}"
echo "Archive checksum: ${RESULT_ARCHIVE}.sha256"
