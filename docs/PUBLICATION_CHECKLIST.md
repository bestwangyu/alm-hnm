# GitHub and Zenodo publication checklist

1. Confirm upstream ALM-DNN redistribution rights and preserve any required third-party notices; the ALM-HNM project license is MIT.
2. Replace the remaining paper title, affiliation, ORCID, repository URL, DOI, and release-date placeholders in `metadata/CITATION.cff.template` and `metadata/zenodo_metadata.template.json`.
3. Extract the unified ZIP and commit the entire release tree to GitHub, including `results/archives/`, `results/audit_logs/`, and `SHA256SUMS`. The ZIP is a local transfer container; do not commit only the outer ZIP. Create a Git tag (for example `v1.0.0`) and publish the corresponding GitHub Release after metadata and licensing are complete.
4. Enable the repository in the GitHub-Zenodo integration before creating the GitHub Release. A push or tag alone is not the release trigger. Let Zenodo archive that same release; do not upload a separately maintained code/data package. Verify the archive contents include all six result bundles and the same checksum manifest. The outer ZIP bytes may differ because GitHub/Zenodo choose their own container names and timestamps; compare the files inside, not the outer ZIP hash.
5. Add the resulting version DOI and GitHub release URL to the data-availability statement and repository README.
6. Re-run `python analysis/verify_release.py` after any file change, regenerate `SHA256SUMS`, and upload checksums with the deposit.
7. Cite the archived software/result package separately from the original ALM paper and from CIFAR/SVHN dataset papers.
8. Do not upload local raw datasets, `.pth` weights, manuscript drafts, author private data, or invalid SVHN v1/v2 artifacts.
