# Release audit

## Completeness decision

This candidate meets the minimum technical standard for code-and-data reproducibility: executable source and configurations are present; formal experiment commands and fixed seed sets are documented; analysis-ready summary and paired tables are included; generated predictions, per-seed metrics, resolved configurations, logs, and source snapshots are retained in six checksum-protected archives; public input datasets have documented access routes; and an offline verifier checks file integrity and evidence boundaries.

## Deliberate exclusions

- Raw CIFAR10, CIFAR100, and SVHN images: third-party public data downloaded by torchvision.
- `.pth` checkpoints: omitted to keep the deposit minimal; retraining commands are supplied.
- SVHN v1/v2 and diagnostic/preflight archives: excluded because they are not valid paper evidence.
- Manuscript drafts, Springer templates, local Skills, temporary inspection folders, and exploratory caches: not needed for computational reproduction.

## Checks completed during assembly

- All six archive sidecar hashes matched.
- Archive members contained no absolute paths, parent traversal, symlinks, hard links, or `.pth` files.
- The 16-row loss comparison reconstructed from all ensemble logs matched the frozen manuscript source table numerically.
- Python source compiled successfully.
- The unified release directory and its full-evidence ZIP are checked with `analysis/verify_release.py`; the ZIP is also checked against its outer sidecar hash and the internal `SHA256SUMS` manifest.

## Residual publication blockers

The ALM-HNM release uses the MIT License, copyright 2026 Wang Yu. The upstream source copy has no LICENSE file, so its public redistribution rights remain unresolved and must be confirmed separately. The final paper title, affiliation, ORCID, repository URL, release date, and Zenodo DOI must also be supplied. The optional plotting script was statically checked but could not be run in the local Windows paper environment because that environment's matplotlib import blocked while building/loading its backend cache; run it in the documented clean analysis environment or on Ubuntu before final publication.
