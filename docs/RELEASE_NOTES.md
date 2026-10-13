# Release notes for 1.0.0-rc1

This is a prepublication reproducibility candidate assembled from the completed local experiment workspace. The original workspace was not modified. This unified package supersedes the earlier code-only GitHub ZIP and separate Zenodo ZIP. The complete repository tree and one Git tag are the single source for both services.

Release-only changes are limited to packaging and reproducibility: a portable environment was added; the CIFAR100 ratio wrapper no longer repeats an archive step already completed by the base runner; the eight-loss summary was reconstructed from all 16 surviving ensemble logs because the mutable summary file had been overwritten by an LDAM-only rerun; and documentation now distinguishes equal-per-source-class evaluation sampling from 1:1 binary balance.

No result values, model definitions, loss functions, or mining rules were changed.
