# Data provenance and access

## Reused public datasets

- CIFAR10 and CIFAR100: torchvision distributions of the datasets introduced by Alex Krizhevsky; downloaded through `torchvision.datasets.CIFAR10` and `CIFAR100` into `datasets/`.
- SVHN: torchvision distribution of the Street View House Numbers dataset; downloaded through `torchvision.datasets.SVHN`.

Raw image files are excluded to avoid duplicate redistribution and to preserve upstream terms. Record the torchvision version and the downloaded file checksums when making the final public deposit.

## Generated research artifacts

The six files under `results/archives/` are immutable copies of completed experiment bundles. They contain prediction logits, seed metrics, resolved configurations, logs, and source snapshots, but no model weights. Their checksums are recorded in both sidecar files and `results/archive_audit.csv`.

SVHN v1 and v2, the SVHN preflight, and the Stage 2 error-diagnosis bundle are excluded because they are diagnostic or invalid evidence. Exploratory parameter-search outputs are not pooled with independent validation results.
