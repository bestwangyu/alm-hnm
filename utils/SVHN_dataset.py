"""Binary imbalanced SVHN dataset wrappers."""

import numpy as np
import torchvision


class _BinarySVHNBase:
    def get_tot_classes_list(self, tot_classes):
        return [self.targets.count(i) for i in range(tot_classes)]


class BinaryUnbalancedSVHN(_BinarySVHNBase, torchvision.datasets.SVHN):
    """Create a binary, imbalanced SVHN split from selected digit labels."""

    def __init__(
        self,
        root,
        imb_factor,
        seed,
        new_majority_labels,
        new_minority_labels,
        tot_classes=2,
        val_samples_per_cls=50,
        perc_per_run=0.9,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        split = "train" if train else "test"
        super(BinaryUnbalancedSVHN, self).__init__(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        img_max = int(
            ((len(self.data) / 10) - val_samples_per_cls) * perc_per_run
        )
        labels = np.asarray(self.labels, dtype=np.int64)
        new_data = []
        source_targets = []

        # SVHN classes have different source sizes. Select the available
        # majority samples first, then derive the minority count from the
        # actual majority total so the requested aggregate ratio is preserved.
        for source_class in new_majority_labels:
            indices = np.where(labels == source_class)[0]
            np.random.seed(seed)
            np.random.shuffle(indices)
            selected = indices[:img_max]
            new_data.append(self.data[selected, ...])
            source_targets.extend([int(source_class)] * len(selected))

        actual_majority = len(source_targets)
        minority_total = max(1, int(actual_majority * imb_factor))
        minority_base, minority_remainder = divmod(
            minority_total, len(new_minority_labels)
        )
        for class_index, source_class in enumerate(new_minority_labels):
            count = minority_base + int(class_index < minority_remainder)
            indices = np.where(labels == source_class)[0]
            np.random.seed(seed)
            np.random.shuffle(indices)
            selected = indices[:count]
            new_data.append(self.data[selected, ...])
            source_targets.extend([int(source_class)] * len(selected))

        self.data = np.concatenate(new_data, axis=0)
        self.targets = [
            0 if target in new_majority_labels else 1
            for target in source_targets
        ]
        self.labels = np.asarray(self.targets, dtype=np.int64)
        if len(self.data) != len(self.targets):
            raise RuntimeError("SVHN data/target length mismatch")
        if np.unique(self.labels).size != tot_classes:
            raise RuntimeError(
                "SVHN split does not contain all {} binary classes".format(
                    tot_classes
                )
            )


class BinaryBalancedSVHN(_BinarySVHNBase, torchvision.datasets.SVHN):
    """Create a balanced binary SVHN validation or test split."""

    def __init__(
        self,
        root,
        tot_classes,
        first_class_labels,
        second_class_labels,
        val_samples_per_cls=50,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        split = "train" if train else "test"
        super(BinaryBalancedSVHN, self).__init__(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        labels = np.asarray(self.labels, dtype=np.int64)
        classes = np.concatenate((first_class_labels, second_class_labels))
        new_data = []
        source_targets = []
        for source_class in classes:
            indices = np.where(labels == source_class)[0]
            np.random.seed(42)
            np.random.shuffle(indices)
            selected = indices[:val_samples_per_cls]
            new_data.append(self.data[selected, ...])
            source_targets.extend([int(source_class)] * val_samples_per_cls)
        self.data = np.concatenate(new_data, axis=0)
        self.targets = [
            0 if target in first_class_labels else 1
            for target in source_targets
        ]
        self.labels = np.asarray(self.targets, dtype=np.int64)
