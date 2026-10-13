# ALM-HNM

ALM-HNM is the reproducibility code and evidence package for the paper on integrating semi-hard top-k hard-negative mining with augmented-Lagrangian training under severe class imbalance.

## Requirements

For training, use Ubuntu 22.04, Python 3.8.5, PyTorch 2.4.1 with CUDA 12.1, torchvision 0.19.1, NumPy 1.19.1, pandas 1.1.3, SciPy 1.5.2, scikit-learn 0.23.2, matplotlib 3.3.2, Pillow 7.2.0, PyYAML 5.4.1, and tqdm 4.50.2. The experiments used two CUDA-capable GPUs. For offline verification and analysis only, Python 3.11 or later with the packages pinned in `environments/environment_analysis.yml` is sufficient.

## Installation

```bash
conda env create -f environments/environment_train.yml
conda activate alm-hnm-train
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

For result verification without retraining:

```bash
conda env create -f environments/environment_analysis.yml
conda activate alm-hnm-analysis
python analysis/verify_release.py
```

## Data preparation

The code uses the public CIFAR10, CIFAR100, and SVHN datasets through `torchvision`. Set `DOWNLOAD=1` in a runner on the first execution. The datasets are stored under the repository-relative `datasets/` directory as `datasets/cifar10/`, `datasets/cifar100/`, and `datasets/svhn/`. No raw images are redistributed in this package. Training uses random 32x32 crops with four-pixel padding and random horizontal flips; validation and test use tensor conversion and dataset-specific normalization.

The binary protocols use CIFAR100 labels `[72, 4, 95, 30, 55]` versus `[73]`, CIFAR10 labels `[0, 2, 3, 4, 5]` versus `[1]`, and SVHN digits `[2, 3, 4, 5, 6]` versus `[1]`. The released construction and imbalance rules are documented in `docs/REPRODUCIBILITY.md` and `docs/DATA_PROVENANCE.md`.

## Usage

```bash
DOWNLOAD=1 GROUP=cifar100_resnet18 bash run_ijmlc_generalization.sh
DOWNLOAD=1 GROUP=cifar10_resnet10 bash run_ijmlc_generalization.sh
DOWNLOAD=1 RUN_TAG=ijmlc_independent_svhn_v3 GROUP=svhn_resnet10 bash run_ijmlc_generalization.sh
DOWNLOAD=1 bash run_ijmlc_cifar100_resnet10_ratio_sensitivity.sh
```

The complete matrix, fixed seeds, parameters, expected outputs, and single-GPU options are in `docs/REPRODUCIBILITY.md`.

## Repository structure

`main.py` is the training and evaluation entry point. `models/` contains the ResNet models, `utils/` contains dataset, loss, training, and ensembling code, and `configs/` stores fixed experiment configurations. The `run_*.sh` files are formal experiment runners. `results/source_data/` contains manuscript-facing tables, `results/summaries/` contains aggregate and paired statistics, `results/audit_logs/` contains retained logs, and `results/archives/` contains six validated result bundles without model weights. `analysis/` contains the offline verifier and optional plotting script. `environments/` contains training and analysis environment specifications.

## Citation

Please cite the associated paper:

> [PAPER TITLE]. International Journal of Machine Learning and Cybernetics. [YEAR]. DOI: [DOI].

After publication, cite the version-specific Zenodo DOI together with the paper. GitHub can generate a software citation from the root `CITATION.cff`; update it with the DOI and final release date after Zenodo creates the record.

## Data and code availability

The code, configurations, source tables, prediction outputs, per-seed metrics, logs, and six result archives are released from one GitHub repository and archived by the linked Zenodo GitHub Release. CIFAR10, CIFAR100, and SVHN are public third-party datasets and remain available from torchvision; they are not copied into this repository. Model checkpoints are omitted, while the fixed scripts, environments, and seeds allow retraining. See `docs/DATA_AVAILABILITY.md`.

## Contact

Wang Yu: wangyu@haou.edu.cn

## License

Copyright (c) 2026 Wang Yu. This project is released under the MIT License; see `LICENSE`. Before public publication, confirm that all upstream ALM-DNN code included in the repository may legally be redistributed. The original local upstream copy did not contain a license file.
