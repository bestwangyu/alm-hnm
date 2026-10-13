"""Validate binary SVHN sampling before launching GPU experiments."""

import argparse

import numpy as np

from utils.SVHN_dataset import BinaryUnbalancedSVHN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/svhn/")
    parser.add_argument("--ratios", type=int, nargs="+", default=[20, 100, 200])
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    majority_labels = np.array([2, 3, 4, 5, 6])
    minority_labels = np.array([1])

    for ratio in args.ratios:
        dataset = BinaryUnbalancedSVHN(
            root=args.root,
            imb_factor=1.0 / ratio,
            seed=args.seed,
            new_majority_labels=majority_labels,
            new_minority_labels=minority_labels,
            val_samples_per_cls=50,
            perc_per_run=0.9,
            train=True,
            download=args.download,
            transform=None,
        )
        targets = np.asarray(dataset.targets, dtype=np.int64)
        labels = np.asarray(dataset.labels, dtype=np.int64)
        if len(dataset.data) != len(targets) or not np.array_equal(targets, labels):
            raise RuntimeError("SVHN data, targets, and labels are not aligned")

        majority = int((targets == 0).sum())
        minority = int((targets == 1).sum())
        if majority == 0 or minority == 0:
            raise RuntimeError("Ratio 1:{} is missing a binary class".format(ratio))

        print(
            "ratio=1:{}, samples={}, majority={}, minority={}, actual_ratio=1:{:.3f}, status=OK".format(
                ratio,
                len(targets),
                majority,
                minority,
                majority / minority,
            )
        )


if __name__ == "__main__":
    main()
