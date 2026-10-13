#!/usr/bin/env bash
set -euo pipefail

# Final convergence validation for the HNM-augmented paper setup.
#
# Purpose:
#   Run only 5 representative configurations, repeated multiple times with
#   different seed blocks, so the final paper conclusion can report both
#   mean and variance.
#
# Representative points:
#   A-FL:
#     - best overall:        k5 / upper=0.60 / random_k=2
#     - low-FPR98 tradeoff:  k5 / upper=0.62 / random_k=2 / delta=0.30 / warmup=7
#   BCE:
#     - balanced best-AUC:   k10 / upper=0.48 / random_k=1 / delta=0.12 / warmup=7
#     - FPR98-focused:      k10 / upper=0.48 / random_k=0 / delta=0.12 / warmup=3
#     - FPR95-focused:      k10 / upper=0.48 / random_k=0 / delta=0.12 / warmup=7
#
# Usage:
#   bash run_final_convergence_validation.sh
#
# Optional overrides:
#   PYTHON_BIN=/path/to/python GPU0_ID=0 GPU1_ID=1 REPEAT_COUNT=3 \
#   SEED_START=1 SEEDS_PER_REPEAT=10 BATCH_SIZE_PER_GPU=128 NUM_WORKERS=8 \
#   bash run_final_convergence_validation.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# GPU assignment
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"

# Repetition control
REPEAT_COUNT="${REPEAT_COUNT:-3}"
SEED_START="${SEED_START:-1}"
SEEDS_PER_REPEAT="${SEEDS_PER_REPEAT:-10}"
SEED_SPLIT="${SEED_SPLIT:-5}"

# Shared experiment setup
DATASET_NAME="${DATASET_NAME:-cifar100}"
RATIO_POS_TRAIN="${RATIO_POS_TRAIN:-200}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"

LOG_ROOT="${LOG_ROOT:-logs/final_convergence_validation}"
mkdir -p "${LOG_ROOT}"
SUMMARY_FILE="${LOG_ROOT}/final_convergence_repeat_summary.csv"
AGG_FILE="${LOG_ROOT}/final_convergence_aggregate_summary.csv"

echo "point_tag,branch_tag,repeat_idx,seed_start,seed_end,baseline_tag,variant,baseline_config,alm_config,delta,hnm_mode,hnm_topk,hnm_random_k,hnm_semi_hard_upper,hnm_warmup_epochs,auc_avg_logits,fpr98,fpr95,fpr90,avg_logits_file" > "${SUMMARY_FILE}"

extract_metric() {
  local pattern="$1"
  local file="$2"
  grep -oP "${pattern}" "${file}" | tail -1 | sed 's/\.$//'
}

build_seed_list() {
  local start="$1"
  local count="$2"
  local end=$((start + count - 1))
  seq -s, "${start}" "${end}"
}

run_dual_gpu_experiment() {
  local point_tag="$1"
  local branch_tag="$2"
  local repeat_idx="$3"
  local baseline_tag="$4"
  local baseline_config="$5"
  local alm_config="$6"
  local delta_value="$7"
  local hnm_mode="$8"
  local hnm_topk="$9"
  local hnm_random_k="${10}"
  local hnm_semi_upper="${11}"
  local hnm_warmup="${12}"
  local variant_base="${13}"
  local seed_start="${14}"

  local seed_end=$((seed_start + SEEDS_PER_REPEAT - 1))
  local seed_split_end=$((seed_start + SEED_SPLIT - 1))
  local seed_right_start=$((seed_split_end + 1))

  local seeds_gpu0
  local seeds_gpu1
  local all_seeds
  seeds_gpu0="$(build_seed_list "${seed_start}" "${SEED_SPLIT}")"
  seeds_gpu1="$(build_seed_list "${seed_right_start}" "$((SEEDS_PER_REPEAT - SEED_SPLIT))")"
  all_seeds="$(build_seed_list "${seed_start}" "${SEEDS_PER_REPEAT}")"

  local variant="${variant_base}_rep${repeat_idx}"
  local model_name="${point_tag}_${variant}"
  local subfolder_name="finalconv_${point_tag}_${variant}_"
  local log_dir="${LOG_ROOT}/${point_tag}/rep${repeat_idx}"
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
    --hnm_warmup_epochs "${hnm_warmup}"
    --hnm_random_k "${hnm_random_k}"
    --hnm_semi_hard_upper "${hnm_semi_upper}"
  )

  echo "============================================================"
  echo "point=${point_tag}, repeat=${repeat_idx}, seed_range=${seed_start}-${seed_end}"
  echo "baseline_config=${baseline_config}, alm_config=${alm_config}, delta=${delta_value}, warmup=${hnm_warmup}"
  echo "hnm_mode=${hnm_mode}, topk=${hnm_topk}, random_k=${hnm_random_k}, semi_upper=${hnm_semi_upper}"
  echo "GPU ${GPU0_ID}: seeds ${seeds_gpu0}"
  echo "GPU ${GPU1_ID}: seeds ${seeds_gpu1}"

  "${PYTHON_BIN}" main.py \
    --train 1 \
    --gpu_id "${GPU0_ID}" \
    --seed_list "${seeds_gpu0}" \
    "${common_args[@]}" \
    > "${log_dir}/train_gpu${GPU0_ID}.log" 2>&1 &
  local pid0=$!

  "${PYTHON_BIN}" main.py \
    --train 1 \
    --gpu_id "${GPU1_ID}" \
    --seed_list "${seeds_gpu1}" \
    "${common_args[@]}" \
    > "${log_dir}/train_gpu${GPU1_ID}.log" 2>&1 &
  local pid1=$!

  wait "${pid0}"
  wait "${pid1}"

  "${PYTHON_BIN}" main.py \
    --train 0 \
    --gpu_id "${GPU0_ID}" \
    --seed_list "${all_seeds}" \
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
  echo "${point_tag},${branch_tag},${repeat_idx},${seed_start},${seed_end},${baseline_tag},${variant},${baseline_config},${alm_config},${delta_value},${hnm_mode},${hnm_topk},${hnm_random_k},${hnm_semi_upper},${hnm_warmup},${auc_avg_logits},${fpr98},${fpr95},${fpr90},${avg_logits_file}" >> "${SUMMARY_FILE}"

  echo "Finished point=${point_tag}, repeat=${repeat_idx}"
  echo "AUC=${auc_avg_logits}, FPR98=${fpr98}, FPR95=${fpr95}, FPR90=${fpr90}"
}

run_point_repeats() {
  local point_tag="$1"
  local branch_tag="$2"
  local baseline_tag="$3"
  local baseline_config="$4"
  local alm_config="$5"
  local delta_value="$6"
  local hnm_mode="$7"
  local hnm_topk="$8"
  local hnm_random_k="$9"
  local hnm_semi_upper="${10}"
  local hnm_warmup="${11}"
  local variant_base="${12}"

  local r
  for ((r=0; r<REPEAT_COUNT; r++)); do
    local seed_start=$((SEED_START + r * SEEDS_PER_REPEAT))
    run_dual_gpu_experiment \
      "${point_tag}" \
      "${branch_tag}" \
      "$((r + 1))" \
      "${baseline_tag}" \
      "${baseline_config}" \
      "${alm_config}" \
      "${delta_value}" \
      "${hnm_mode}" \
      "${hnm_topk}" \
      "${hnm_random_k}" \
      "${hnm_semi_upper}" \
      "${hnm_warmup}" \
      "${variant_base}" \
      "${seed_start}"
  done
}

# Representative points
run_point_repeats \
  "a_fl_best_auc" \
  "a_fl" \
  "a_fl" \
  "baseline_a_FL_g2" \
  "ALM_13_beg" \
  "0.25" \
  "semi_hard_topk" \
  "5" \
  "2" \
  "0.60" \
  "5" \
  "k5_u0p60_r2"

run_point_repeats \
  "a_fl_low_fpr98" \
  "a_fl" \
  "a_fl" \
  "baseline_a_FL_g2" \
  "ALM_13_beg" \
  "0.30" \
  "semi_hard_topk" \
  "5" \
  "2" \
  "0.62" \
  "7" \
  "k5_u0p62_r2_d0p30_w7"

run_point_repeats \
  "bce_best_auc" \
  "bce" \
  "bce" \
  "baseline_bce" \
  "ALM_13_beg" \
  "0.12" \
  "semi_hard_topk" \
  "10" \
  "1" \
  "0.48" \
  "7" \
  "k10_u0p48_r1_d0p12_w7"

run_point_repeats \
  "bce_low_fpr98" \
  "bce" \
  "bce" \
  "baseline_bce" \
  "ALM_13_beg" \
  "0.12" \
  "semi_hard_topk" \
  "10" \
  "0" \
  "0.48" \
  "3" \
  "k10_u0p48_r0_d0p12_w3"

run_point_repeats \
  "bce_low_fpr95" \
  "bce" \
  "bce" \
  "baseline_bce" \
  "ALM_13_beg" \
  "0.12" \
  "semi_hard_topk" \
  "10" \
  "0" \
  "0.48" \
  "7" \
  "k10_u0p48_r0_d0p12_w7"

{
  echo "point_tag,branch_tag,baseline_tag,baseline_config,alm_config,delta,hnm_mode,hnm_topk,hnm_random_k,hnm_semi_hard_upper,hnm_warmup_epochs,n,auc_mean,auc_std,fpr98_mean,fpr98_std,fpr95_mean,fpr95_std,fpr90_mean,fpr90_std"
  awk -F, '
    NR==1 { next }
    {
      k = $1
      if (!(k in seen)) {
        seen[k] = 1
        branch[k] = $2
        baseline[k] = $6
        baseline_cfg[k] = $8
        alm_cfg[k] = $9
        delta[k] = $10
        mode[k] = $11
        topk[k] = $12
        random_k[k] = $13
        upper[k] = $14
        warmup[k] = $15
      }
      n[k]++
      auc[k] += $16; auc2[k] += $16 * $16
      f98[k] += $17; f982[k] += $17 * $17
      f95[k] += $18; f952[k] += $18 * $18
      f90[k] += $19; f902[k] += $19 * $19
    }
    END {
      for (k in n) {
        auc_mean = auc[k] / n[k]
        f98_mean = f98[k] / n[k]
        f95_mean = f95[k] / n[k]
        f90_mean = f90[k] / n[k]
        auc_std = sqrt(auc2[k] / n[k] - auc_mean * auc_mean)
        f98_std = sqrt(f982[k] / n[k] - f98_mean * f98_mean)
        f95_std = sqrt(f952[k] / n[k] - f95_mean * f95_mean)
        f90_std = sqrt(f902[k] / n[k] - f90_mean * f90_mean)
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", \
          k, branch[k], baseline[k], baseline_cfg[k], alm_cfg[k], delta[k], mode[k], topk[k], random_k[k], upper[k], warmup[k], n[k], \
          auc_mean, auc_std, f98_mean, f98_std, f95_mean, f95_std, f90_mean, f90_std
      }
    }
  ' "${SUMMARY_FILE}" | sort
} > "${AGG_FILE}"

echo "============================================================"
echo "Final convergence validation finished."
echo "Repeat summary: ${SUMMARY_FILE}"
echo "Aggregate summary: ${AGG_FILE}"
