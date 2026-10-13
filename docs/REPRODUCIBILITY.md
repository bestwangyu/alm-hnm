# Reproducibility protocol

## Formal experiment matrix

| Evidence family | Dataset / backbone | Ratio | Seeds | Runner | Valid archive |
|---|---|---:|---|---|---|
| mining ablation | CIFAR100 / ResNet10 | 1:200 | 31-50 | `run_ijmlc_ablation.sh` | `ijmlc_ablation_v1_results_no_pth.tar.gz` |
| parameter robustness | CIFAR100 / ResNet10 | 1:200 | 51-70 | `run_ijmlc_stage2_robustness.sh` | `ijmlc_stage2_robustness_v2_results_no_pth.tar.gz` |
| ratio generalization | CIFAR10 / ResNet10 | 1:20, 1:100, 1:200 | 71-90 | `run_ijmlc_generalization.sh` | `ijmlc_generalization_v1_cifar10_resnet10_results_no_pth.tar.gz` |
| backbone generalization | CIFAR100 / ResNet18 | 1:200 | 71-90 | `run_ijmlc_generalization.sh` | `ijmlc_generalization_v1_cifar100_resnet18_results_no_pth.tar.gz` |
| independent dataset | SVHN / ResNet10 | 1:20, 1:100, 1:200 | 71-90 | `run_ijmlc_generalization.sh` | `ijmlc_independent_svhn_v3_svhn_resnet10_results_no_pth.tar.gz` |
| ratio sensitivity | CIFAR100 / ResNet10 | 1:20, 1:100 | 71-90 | `run_ijmlc_cifar100_resnet10_ratio_sensitivity.sh` | `ijmlc_cifar100_resnet10_ratio_v1_cifar100_resnet10_results_no_pth.tar.gz` |
| loss heterogeneity | CIFAR100 / ResNet10 | 1:200 | 1-10 | `run_baseline_compare_semihard.sh` | reconstructed table plus 16 ensemble logs |
| convergence validation | CIFAR100 / ResNet10 | 1:200 | three disjoint 10-seed blocks | `run_final_convergence_validation.sh` | aggregate/repeat tables plus logs |

The mining ablation compares `all`, `random`, `topk`, `semi_hard`, and `semi_hard_topk`. The fixed mixed setting for generalization is top-k 20, upper semi-hard bound 0.30, warm-up 5 epochs, random-k 0, with clipping values 10, 1, and 5. Stage 2 evaluates a predefined top-k/upper-bound grid and two random-k variants.

## Data construction

CIFAR100 uses source classes `[72, 4, 95, 30, 55]` as binary class 0 and source class `[73]` as binary class 1. CIFAR10 uses `[0, 2, 3, 4, 5]` versus `[1]`; SVHN uses digits `[2, 3, 4, 5, 6]` versus `[1]`. Validation selects 50 examples per source class and test selects 100 per source class, so their aggregate binary class counts are 250:50 and 500:100. The word “balanced” in wrapper class names refers to equal sampling per source class, not a 1:1 aggregate binary split.

For CIFAR, each majority source class contributes 405 training examples. The minority count is computed from the requested aggregate ratio and integer-truncated. SVHN v3 first records the available majority total and derives the minority count from that total, correcting the invalid v1/v2 construction.

## Determinism and hardware

Each run seeds Python, NumPy, and PyTorch and enables deterministic cuDNN behavior. Hardware/library differences can still cause small floating-point deviations. Preserve the fixed seed lists and compare distributions and operating-point metrics rather than requiring bitwise-identical model weights.

## Single-GPU fallback

The generalization runner supports `SINGLE_GPU=1 SINGLE_GPU_ID=0`. Other formal runners are designed for two GPUs and split seed sets across processes. Batch size is per training process; the recorded value is 128 and `NUM_WORKERS=8` per process.

## Expected output

Formal runners produce per-seed metrics, prediction logits, resolved configuration snapshots, ensemble logs, aggregate/paired CSVs, a source snapshot, `RESULT_MANIFEST.txt`, a `tar.gz` excluding `.pth`, and a SHA-256 sidecar. Always retain the master log and checksum with transferred results.
