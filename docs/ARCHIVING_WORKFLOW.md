# Single-source GitHub-Zenodo archive

Use the full unified directory for GitHub and Zenodo. Commit the six small result tarballs and the supporting logs as normal repository files, not Git LFS pointers or release-only attachments. They total only a few megabytes and should be part of the tagged source tree.

Complete authorship and license metadata, regenerate the manifest if files change, and verify the directory before committing. Enable the repository in Zenodo, create one GitHub tag and its Release, then inspect the Zenodo record to confirm the tag and internal file checksums. Publishing a GitHub commit alone does not trigger the normal release archive workflow.

After DOI creation, use a later commit to add the DOI to the README, or reserve the DOI beforehand if your workflow supports it. Do not move an existing published tag to include the DOI. A DOI added in a subsequent repository commit does not change the immutable tagged archive.

The local unified ZIP is only a transfer container. Extract and commit its contents. Zenodo's generated ZIP need not have the same outer hash; the files at the archived tag must match the internal `SHA256SUMS`.
