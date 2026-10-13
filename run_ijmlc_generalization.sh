#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
SEEDS_GPU0="${SEEDS_GPU0:-71,72,73,74,75,76,77,78,79,80}"
SEEDS_GPU1="${SEEDS_GPU1:-81,82,83,84,85,86,87,88,89,90}"
ALL_SEEDS="${ALL_SEEDS:-71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90}"
FIRST_SEED="${ALL_SEEDS%%,*}"
LAST_SEED="${ALL_SEEDS##*,}"
SEED_RANGE_TAG="seeds${FIRST_SEED}-${LAST_SEED}"
GROUP="${GROUP:-cifar100_resnet18}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
DOWNLOAD="${DOWNLOAD:-0}"
RUN_TAG="${RUN_TAG:-ijmlc_generalization_v1}"
SINGLE_GPU="${SINGLE_GPU:-0}"
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"

case "${GROUP}" in
  cifar100_resnet18)
    DATASET_NAME=cifar100; RESNET_TYPE=resnet18; ALM_CONFIG=ALM_13_beg
    DELTA=0.12; RATIOS="${RATIOS:-200}" ;;
  cifar100_resnet10)
    DATASET_NAME=cifar100; RESNET_TYPE=resnet10; ALM_CONFIG=ALM_13_beg
    DELTA=0.10; RATIOS="${RATIOS:-20 100}" ;;
  cifar10_resnet10)
    DATASET_NAME=cifar10; RESNET_TYPE=resnet10; ALM_CONFIG=ALM_15_beg
    DELTA=0.5; RATIOS="${RATIOS:-20 100 200}" ;;
  svhn_resnet10)
    DATASET_NAME=svhn; RESNET_TYPE=resnet10; ALM_CONFIG=ALM_13_beg
    DELTA=0.12; RATIOS="${RATIOS:-20 100 200}" ;;
  *) echo "GROUP must be cifar100_resnet18, cifar100_resnet10, cifar10_resnet10, or svhn_resnet10" >&2; exit 2 ;;
esac

LOG_ROOT="logs/${RUN_TAG}/${GROUP}"
mkdir -p "${LOG_ROOT}"

if [[ "${SINGLE_GPU}" == "1" ]]; then
  GPU0_ID="${SINGLE_GPU_ID}"
  SEEDS_GPU0="${ALL_SEEDS}"
  SEEDS_GPU1=""
fi

if [[ "${DOWNLOAD}" == "1" ]]; then
  "${PYTHON_BIN}" - "${DATASET_NAME}" <<'PY'
import sys
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
name = sys.argv[1]
root = "datasets/" + name
if name == "cifar10":
    CIFAR10(root=root, train=True, download=True)
    CIFAR10(root=root, train=False, download=True)
elif name == "cifar100":
    CIFAR100(root=root, train=True, download=True)
    CIFAR100(root=root, train=False, download=True)
else:
    SVHN(root=root, split="train", download=True)
    SVHN(root=root, split="test", download=True)
PY
fi

run_one() {
  local ratio="$1" method="$2" mode="$3" topk="$4" upper="$5" warmup="$6"
  local label="r${ratio}_${method}"
  local model="${RUN_TAG}_${GROUP}_${label}"
  local subfolder="${RUN_TAG}_${GROUP}_${label}_"
  local logdir="${LOG_ROOT}/${label}"
  mkdir -p "${logdir}"
  local args=(
    --dataset_name "${DATASET_NAME}" --name_file_bs baseline_bce
    --name_file_alm "${ALM_CONFIG}" --model_name "${model}"
    --subfolder_name "${subfolder}" --ratio_pos_train "${ratio}"
    --delta "${DELTA}" --resnet_type "${RESNET_TYPE}" --download 0
    --bsize "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}"
    --num_epochs "${NUM_EPOCHS}" --hnm_mode "${mode}"
    --hnm_topk "${topk}" --hnm_semi_hard_upper "${upper}"
    --hnm_warmup_epochs "${warmup}" --hnm_random_k 0
    --alm_q_clip 10.0 --alm_lambda_clip 1.0 --grad_clip_norm 5.0
  )
  echo "Running ${GROUP}, ratio=1:${ratio}, method=${method}"
  set +e
  "${PYTHON_BIN}" main.py --train 1 --gpu_id "${GPU0_ID}" \
    --seed_list "${SEEDS_GPU0}" "${args[@]}" > "${logdir}/train_gpu${GPU0_ID}.log" 2>&1 &
  local p0=$!
  local p1=""
  if [[ "${SINGLE_GPU}" != "1" ]]; then
    "${PYTHON_BIN}" main.py --train 1 --gpu_id "${GPU1_ID}" \
      --seed_list "${SEEDS_GPU1}" "${args[@]}" > "${logdir}/train_gpu${GPU1_ID}.log" 2>&1 &
    p1=$!
  fi
  wait "${p0}"; local s0=$?
  local s1=0
  if [[ -n "${p1}" ]]; then wait "${p1}"; s1=$?; fi
  set -e
  [[ "${s0}" == 0 && "${s1}" == 0 ]] || {
    echo "Training failed: ${label}, GPU${GPU0_ID}=${s0}, GPU${GPU1_ID}=${s1}" >&2
    return 1
  }
  "${PYTHON_BIN}" main.py --train 0 --gpu_id "${GPU0_ID}" \
    --seed_list "${ALL_SEEDS}" "${args[@]}" > "${logdir}/ensemble.log" 2>&1
}

for ratio in ${RATIOS}; do
  run_one "${ratio}" all all 0 0 0
  run_one "${ratio}" semi_hard_topk semi_hard_topk 20 0.30 5
done

SUMMARY="${LOG_ROOT}/${RUN_TAG}_${GROUP}_summary.csv"
PAIRED="${LOG_ROOT}/${RUN_TAG}_${GROUP}_paired.csv"
"${PYTHON_BIN}" - "${SUMMARY}" "${PAIRED}" "${GROUP}" "${DATASET_NAME}" "${RESNET_TYPE}" "${RATIOS}" "${RUN_TAG}" "${SEED_RANGE_TAG}" <<'PY'
import csv
import os
import sys
import numpy as np
from scipy import stats

summary, paired, group, dataset, resnet, ratios_text, run_tag, seed_range_tag = sys.argv[1:]
rows_out = []
paired_out = []
for ratio in ratios_text.split():
    ratio_label = "r" + ratio
    root = os.path.join("logs", run_tag, group)
    # The run tag is encoded in the supplied summary path; locate checkpoints
    # from the deterministic names used by run_one.
    for method in ("all", "semi_hard_topk"):
        model = "{}_{}_{}_{}".format(run_tag, group, ratio_label, method)
        checkpoint = os.path.join(
            "checkpoint", "{}_{}_{}_{}_".format(run_tag, group, ratio_label, method), model
        )
        seed_file = os.path.join(checkpoint, model + "_eval_seed_metrics_" + seed_range_tag + ".csv")
        ensemble_file = os.path.join(checkpoint, model + "_eval_ensemble_metrics_" + seed_range_tag + ".csv")
        with open(seed_file, newline="") as handle:
            seed_rows = list(csv.DictReader(handle))
        with open(ensemble_file, newline="") as handle:
            ensemble = next(csv.DictReader(handle))
        def values(name):
            return np.asarray([float(row[name]) for row in seed_rows], dtype=float)
        names = {
            "validation_auc_max": "validation_auc",
            "test_roc_auc": "test_roc_auc",
            "test_auprc": "test_auprc",
            "test_fpr_at_tpr90": "fpr90",
            "test_fpr_at_tpr95": "fpr95",
            "test_fpr_at_tpr98": "fpr98",
        }
        out = {"group": group, "dataset": dataset, "resnet": resnet,
               "ratio": ratio, "method": method, "n": len(seed_rows)}
        for source, target in names.items():
            v = values(source)
            out[target + "_mean"] = v.mean()
            out[target + "_std"] = v.std(ddof=1)
        out.update({
            "ensemble_roc_auc": ensemble["ensemble_roc_auc"],
            "ensemble_auprc": ensemble["ensemble_auprc"],
            "ensemble_fpr90": ensemble["ensemble_fpr_at_tpr90"],
            "ensemble_fpr95": ensemble["ensemble_fpr_at_tpr95"],
            "ensemble_fpr98": ensemble["ensemble_fpr_at_tpr98"],
        })
        rows_out.append(out)
    control = {int(r["seed"]): r for r in seed_rows if False}
    files = {}
    for method in ("all", "semi_hard_topk"):
        model = "{}_{}_{}_{}".format(run_tag, group, ratio_label, method)
        checkpoint = os.path.join(
            "checkpoint", "{}_{}_{}_{}_".format(run_tag, group, ratio_label, method), model
        )
        with open(os.path.join(checkpoint, model + "_eval_seed_metrics_" + seed_range_tag + ".csv"), newline="") as handle:
            files[method] = {int(r["seed"]): r for r in csv.DictReader(handle)}
    for metric in ("test_roc_auc", "test_auprc", "test_fpr_at_tpr90", "test_fpr_at_tpr95", "test_fpr_at_tpr98"):
        shared = sorted(set(files["all"]) & set(files["semi_hard_topk"]))
        if not shared:
            raise SystemExit("No shared seeds for {} at ratio 1:{}".format(metric, ratio))
        d = np.asarray([float(files["all"][s][metric]) - float(files["semi_hard_topk"][s][metric]) for s in shared])
        if not np.all(np.isfinite(d)):
            raise SystemExit("Non-finite paired differences for {} at ratio 1:{}".format(metric, ratio))
        boot = d[np.random.RandomState(20260902).randint(0, len(d), size=(10000, len(d)))].mean(axis=1)
        # SciPy cannot apply the Wilcoxon test when every paired difference is zero.
        # In that case the observed effect is exactly zero, so report p=1.0.
        if np.allclose(d, 0.0):
            wilcoxon_pvalue = 1.0
        else:
            try:
                wilcoxon_pvalue = float(stats.wilcoxon(d).pvalue)
            except ValueError:
                wilcoxon_pvalue = 1.0
        paired_out.append({
            "group": group, "dataset": dataset, "resnet": resnet, "ratio": ratio,
            "metric": metric, "n": len(d), "control_minus_method_mean": d.mean(),
            "ci95_low": np.percentile(boot, 2.5), "ci95_high": np.percentile(boot, 97.5),
            "wilcoxon_pvalue": wilcoxon_pvalue,
        })

fields = ["group", "dataset", "resnet", "ratio", "method", "n"]
for name in ("validation_auc", "test_roc_auc", "test_auprc"):
    fields += [name + "_mean", name + "_std"]
fields += ["fpr90_mean", "fpr90_std", "fpr95_mean", "fpr95_std", "fpr98_mean", "fpr98_std",
           "ensemble_roc_auc", "ensemble_auprc", "ensemble_fpr90", "ensemble_fpr95", "ensemble_fpr98"]
with open(summary, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows_out)
with open(paired, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(paired_out[0]))
    writer.writeheader()
    writer.writerows(paired_out)
PY

mkdir -p "${LOG_ROOT}/source_snapshot/configs/baseline" "${LOG_ROOT}/source_snapshot/configs/ALM"
cp main.py utils/custom_losses.py utils/train_test_CIFAR.py \
  utils/final_ensembling_results.py utils/CIFAR100_dataset.py \
  utils/SVHN_dataset.py check_svhn_dataset.py environment.yml \
  "${LOG_ROOT}/source_snapshot/"
cp configs/baseline/baseline_bce.json "${LOG_ROOT}/source_snapshot/configs/baseline/"
cp "configs/ALM/${ALM_CONFIG}.json" "${LOG_ROOT}/source_snapshot/configs/ALM/"
cp "${BASH_SOURCE[0]}" "${LOG_ROOT}/source_snapshot/"
pushd "${LOG_ROOT}/source_snapshot" >/dev/null
find . -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS
popd >/dev/null
{
  uname -a; "${PYTHON_BIN}" --version; "${PYTHON_BIN}" -m pip freeze
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} > "${LOG_ROOT}/environment.txt" 2>&1

RESULT_ARCHIVE="${RUN_TAG}_${GROUP}_results_no_pth.tar.gz"
checkpoint_paths=()
for ratio in ${RATIOS}; do
  for method in all semi_hard_topk; do
    checkpoint_paths+=("checkpoint/${RUN_TAG}_${GROUP}_r${ratio}_${method}_")
  done
done
find "${LOG_ROOT}" "${checkpoint_paths[@]}" -type f ! -name '*.pth' -printf '%p\t%s bytes\n' \
  | sort > "${LOG_ROOT}/RESULT_MANIFEST.txt"
tar --exclude='*.pth' -czf "${RESULT_ARCHIVE}" "${LOG_ROOT}" "${checkpoint_paths[@]}"
sha256sum "${RESULT_ARCHIVE}" > "${RESULT_ARCHIVE}.sha256"
echo "Archive: ${RESULT_ARCHIVE}"
