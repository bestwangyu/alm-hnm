# Data dictionary

## Core metric fields

| Field | Meaning | Scale / direction |
|---|---|---|
| `validation_auc_max` or `validation_auc_mean` | maximum validation ROC-AUC used during a run | 0-1, higher is better |
| `test_roc_auc` | per-seed test ROC-AUC | 0-1, higher is better |
| `test_auprc` | per-seed test area under the precision-recall curve | 0-1, higher is better |
| `test_fpr_at_tpr90` / `fpr90` | false-positive rate at TPR 0.90 | 0-1, lower is better |
| `test_fpr_at_tpr95` / `fpr95` | false-positive rate at TPR 0.95 | 0-1, lower is better |
| `test_fpr_at_tpr98` / `fpr98` | false-positive rate at TPR 0.98 | 0-1, lower is better |
| `ensemble_*` | metric from logits averaged over the stated seed set | 0-1 |
| `*_percent` | metric multiplied by 100 | percentage points |
| `n` | number of paired seeds, except convergence where it is seed-block repeats | integer |

## Paired-effect fields

Generalization tables define `control_minus_method_mean` as Original ALM minus semi-hard top-k. Negative values favor mining for ROC-AUC/AUPRC; positive values favor mining for FPR. The included Wilcoxon p-values are raw unless a separate corrected column is explicitly present. Do not describe them as multiplicity-corrected.

## Result levels

`seed_metrics` contains per-seed observations. `ensemble_summary` describes one aggregate prediction obtained by averaging logits across a seed set and must not be treated as multiple independent observations. `figure6_convergence_aggregate.csv` has three disjoint seed-block repeats per setting. `source_data` contains frozen manuscript-facing tables derived from the named evidence files.
