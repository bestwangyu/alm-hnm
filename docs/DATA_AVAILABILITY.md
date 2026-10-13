# Data and code availability

## Public input datasets

This study uses CIFAR10, CIFAR100, and the Street View House Numbers (SVHN) dataset. CIFAR10 and CIFAR100 are available from the official CIFAR page at https://www.cs.toronto.edu/~kriz/cifar.html and through `torchvision.datasets.CIFAR10` and `torchvision.datasets.CIFAR100`. SVHN is available from http://ufldl.stanford.edu/housenumbers/ and through `torchvision.datasets.SVHN`.

The raw third-party images are not redistributed. On first execution, set `DOWNLOAD=1`; torchvision downloads them to the repository-relative locations `datasets/cifar10/`, `datasets/cifar100/`, and `datasets/svhn/`. These directories are excluded from Git. Binary label mappings, imbalance construction, validation/test sampling, and preprocessing are described in `docs/REPRODUCIBILITY.md` and implemented in `main.py`, `utils/CIFAR100_dataset.py`, and `utils/SVHN_dataset.py`.

## Data included in this repository

Analysis-ready manuscript tables are under `results/source_data/`. Aggregate, paired, and per-seed statistical summaries are under `results/summaries/`. Retained baseline and convergence logs are under `results/audit_logs/`. Six checksum-protected experiment bundles are under `results/archives/`; they contain prediction logits, per-seed metrics, resolved configurations, source snapshots, and execution logs. Their provenance and validity are indexed by `results/EVIDENCE_INDEX.csv` and `results/archive_audit.csv`.

Model checkpoint files (`.pth`) are not included. The reported models can be retrained using the released scripts, fixed configurations, random seeds, and environment specifications. Run `python analysis/verify_release.py` to verify the released evidence and checksums.

## Manuscript statement

The code, fixed configurations, analysis-ready source data, prediction outputs, per-seed metrics, and execution logs supporting this study are available at [GITHUB RELEASE URL] and are archived from the same GitHub Release through Zenodo at [VERSION DOI]. CIFAR10, CIFAR100, and SVHN are publicly available third-party datasets and are not redistributed; their official access routes and the exact class-construction protocol are documented in the repository. Model checkpoints are not included, but all reported models can be retrained from the released scripts, environments, fixed seeds, and configurations.

Use the Zenodo version DOI for the submitted manuscript. Use the Zenodo concept DOI only when referring to the evolving ALM-HNM project across releases.
