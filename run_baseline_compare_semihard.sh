#!/usr/bin/env bash
set -euo pipefail

# Full baseline comparison under cifar100 / ratio 1:200.
# For each baseline, this script runs:
#   1) ALM + baseline (control, no mining)
#   2) ALM + baseline + semi_hard_topk (fixed best mining params)
#
# It uses two GPUs by splitting seeds into two groups for every experiment.
#
# Usage:
#   bash run_baseline_compare_semihard.sh
#
# Optional overrides:
#   PYTHON_BIN=/path/to/python GPU0_ID=0 GPU1_ID=1 BATCH_SIZE_PER_GPU=128 NUM_WORKERS=8 \
#   bash run_baseline_compare_semihard.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# GPU assignment
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"

# Seed split
SEEDS_GPU0="${SEEDS_GPU0:-1,2,3,4,5}"
SEEDS_GPU1="${SEEDS_GPU1:-6,7,8,9,10}"
ALL_SEEDS="${ALL_SEEDS:-1,2,3,4,5,6,7,8,9,10}"

# Shared experiment setup
DATASET_NAME="${DATASET_NAME:-cifar100}"
RATIO_POS_TRAIN="${RATIO_POS_TRAIN:-200}"

# Throughput defaults
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Fixed best mining params from your sweep
MINING_MODE="${MINING_MODE:-semi_hard_topk}"
MINING_TOPK="${MINING_TOPK:-5}"
MINING_WARMUP="${MINING_WARMUP:-5}"
MINING_RANDOM_K="${MINING_RANDOM_K:-2}"
MINING_SEMI_UPPER="${MINING_SEMI_UPPER:-0.5}"

# Which groups to run
RUN_CONTROL="${RUN_CONTROL:-1}"
RUN_MINING="${RUN_MINING:-1}"
BASELINE_FILTER="${BASELINE_FILTER:-}"

# Logging and summary
LOG_ROOT="${LOG_ROOT:-logs/baseline_compare}"
mkdir -p "${LOG_ROOT}"
SUMMARY_FILE="${LOG_ROOT}/baseline_compare_summary.csv"
echo "baseline_tag,variant,baseline_config,alm_config,delta,hnm_mode,hnm_topk,hnm_semi_hard_upper,auc_avg_logits,fpr98,fpr95,fpr90,avg_logits_file" > "${SUMMARY_FILE}"

# cifar100 / ratio 1:200 settings from the paper README, mapped to actual local filenames
BASELINES=(
  "bce|baseline_bce|ALM_13_beg|0.1"
  "s_lm|baseline_s_LM_m2|ALM_14_beg|0.5"
  "s_fl|baseline_s_FL_g2|ALM_13_beg|0.1"
  "a_lm|baseline_a_LM_m4|ALM_13_beg|0.5"
  "a_fl|baseline_a_FL_g2|ALM_13_beg|0.25"
  "cb_bce|baseline_cb_BCE_99|ALM_13_beg|1.0"
  "w_bce|baseline_WBCE_1|ALM_16_beg|1.0"
  "ldam|baseline_LDAM|ALM_13_beg|0.25"
)

extract_metric() {
  local pattern="$1"
  local file="$2"
  grep -oP "${pattern}" "${file}" | tail -1 | sed 's/\.$//'
}

run_dual_gpu_experiment() {
  local baseline_tag="$1"
  local baseline_config="$2"
  local alm_config="$3"
  local delta_value="$4"
  local variant="$5"
  local hnm_mode="$6"
  local hnm_topk="$7"
  local hnm_semi_upper="$8"

  local model_name="${baseline_tag}_${variant}"
  local subfolder_name="cmp_${baseline_tag}_${variant}_"
  local log_dir="${LOG_ROOT}/${baseline_tag}/${variant}"
  mkdir -p "${log_dir}"

  local common_args=(
    --dataset_name "${DATASET_NAME}"
    --name_file_bs "${baseline_config}"
    --name_file_alm "${alm_config}"
    --model_name "${model_name}"
    --subfolder_name "${subfolder_name}"
    --ratio_pos_train "${RATIO_POS_TRAIN}"
    --delta "${delta_value}"
    --bsize "${BATCH_SIZE_PER_GPU}"
    --num_workers "${NUM_WORKERS}"
    --hnm_mode "${hnm_mode}"
    --hnm_topk "${hnm_topk}"
    --hnm_warmup_epochs "${MINING_WARMUP}"
    --hnm_random_k "${MINING_RANDOM_K}"
    --hnm_semi_hard_upper "${hnm_semi_upper}"
  )

  echo "============================================================"
  echo "baseline=${baseline_tag}, variant=${variant}"
  echo "baseline_config=${baseline_config}, alm_config=${alm_config}, delta=${delta_value}"
  echo "hnm_mode=${hnm_mode}, topk=${hnm_topk}, semi_upper=${hnm_semi_upper}"
  echo "GPU ${GPU0_ID}: seeds ${SEEDS_GPU0}"
  echo "GPU ${GPU1_ID}: seeds ${SEEDS_GPU1}"
  echo "bsize=${BATCH_SIZE_PER_GPU}, num_workers=${NUM_WORKERS}"

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
  wait "${pid1}"

  "${PYTHON_BIN}" main.py \
    --train 0 \
    --gpu_id "${GPU0_ID}" \
    --seed_list "${ALL_SEEDS}" \
    "${common_args[@]}" \
    > "${log_dir}/ensemble.log" 2>&1

  local auc_avg_logits
  local fpr98
  local fpr95
  local fpr90
  auc_avg_logits="$(extract_metric 'AUC of averaged logits of final test: \K[0-9.]+' "${log_dir}/ensemble.log")"
  fpr98="$(extract_metric 'TPR = 98.0, FPR = \K[0-9.]+' "${log_dir}/ensemble.log")"
  fpr95="$(extract_metric 'TPR = 95.0, FPR = \K[0-9.]+' "${log_dir}/ensemble.log")"
  fpr90="$(extract_metric 'TPR = 90.0, FPR = \K[0-9.]+' "${log_dir}/ensemble.log")"

  local avg_logits_file="checkpoint/${subfolder_name}/${model_name}/${model_name}_avg_logits_fv.csv"
  echo "${baseline_tag},${variant},${baseline_config},${alm_config},${delta_value},${hnm_mode},${hnm_topk},${hnm_semi_upper},${auc_avg_logits},${fpr98},${fpr95},${fpr90},${avg_logits_file}" >> "${SUMMARY_FILE}"

  echo "Finished baseline=${baseline_tag}, variant=${variant}"
  echo "AUC=${auc_avg_logits}, FPR98=${fpr98}, FPR95=${fpr95}, FPR90=${fpr90}"
}

for row in "${BASELINES[@]}"; do
  IFS='|' read -r baseline_tag baseline_config alm_config delta_value <<< "${row}"

  if [[ -n "${BASELINE_FILTER}" && "${baseline_tag}" != "${BASELINE_FILTER}" ]]; then
    continue
  fi

  if [[ "${RUN_CONTROL}" == "1" ]]; then
    run_dual_gpu_experiment \
      "${baseline_tag}" \
      "${baseline_config}" \
      "${alm_config}" \
      "${delta_value}" \
      "alm_only" \
      "all" \
      "${MINING_TOPK}" \
      "${MINING_SEMI_UPPER}"
  fi

  if [[ "${RUN_MINING}" == "1" ]]; then
    run_dual_gpu_experiment \
      "${baseline_tag}" \
      "${baseline_config}" \
      "${alm_config}" \
      "${delta_value}" \
      "alm_plus_semihard" \
      "${MINING_MODE}" \
      "${MINING_TOPK}" \
      "${MINING_SEMI_UPPER}"
  fi
done

echo "============================================================"
echo "Baseline comparison finished."
echo "Summary file: ${SUMMARY_FILE}"
