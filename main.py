""" Main to run training and test """
import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import warnings
import torch
from torchvision import transforms
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from models.resnet import ResNet10, ResNet18
from utils.train_test_CIFAR import train, test
from utils.CIFAR100_dataset import (
    BinaryUnbalancedCIFAR10,
    BinaryBalancedCIFAR10,
    BinaryUnbalancedCIFAR100,
    BinaryBalancedCIFAR100,
)
from utils.SVHN_dataset import BinaryUnbalancedSVHN, BinaryBalancedSVHN
from utils.final_ensembling_results import (
    final_ensembling_results,
    validation_ensembling,
)


def parse_seed_list(seed_list_arg):
    """
    Parse a comma-separated seed list such as "1,2,3,4,5".
    """
    return [int(seed.strip()) for seed in seed_list_arg.split(",") if seed.strip()]


def fpr_at_tpr(labels, scores, target_tpr):
    """Return the lowest observed FPR that reaches the requested TPR."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eligible_fpr = fpr[tpr >= target_tpr]
    return float(np.min(eligible_fpr)) if eligible_fpr.size else np.nan

########################################
# Set parameters
########################################

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=int, default=1, help="whether to run also train")
parser.add_argument(
    "--name_file_bs", type=str, default="", help="json file of the configuration"
)
parser.add_argument(
    "--name_file_alm", type=str, default="", help="json file of the configuration"
)
parser.add_argument("--model_name", type=str, default="prova", help="model name")
parser.add_argument(
    "--subfolder_name", type=str, default="prova", help="subfolder name"
)
parser.add_argument(
    "--gpu_id", type=int, default=0, help="GPU id to use for this process"
)
parser.add_argument(
    "--seed_list",
    type=str,
    default="1,2,3,4,5,6,7,8,9,10",
    help="comma-separated list of seeds to process in this run",
)
parser.add_argument(
    "--dataset_name", type=str, default="cifar10", help="subfolder name"
)
parser.add_argument(
    "--download", type=int, default=0, help="download a missing torchvision dataset"
)
parser.add_argument("--bsize", type=int, default=64, help="set batch size")
parser.add_argument(
    "--num_workers", type=int, default=4, help="number of dataloader workers"
)
parser.add_argument("--lr_decrease", type=int, default=0, help="decrease learning rate")
parser.add_argument(
    "--patience_lr",
    type=int,
    default=10,
    help="change lr at AUC plateau after patience",
)
parser.add_argument(
    "--patience_early_stopping",
    type=int,
    default=30,
    help="after this patience do early stopping",
)
parser.add_argument(
    "--continue_train",
    type=int,
    default=0,
    help="continue training from a pretrained model",
)
parser.add_argument(
    "--continue_train_epochs",
    type=int,
    default=50,
    help="how long continuing training for",
)
parser.add_argument(
    "--baseline_cont_train",
    type=str,
    default="prova",
    help="name of the baseline to continue the training",
)
parser.add_argument(
    "--ratio_pos_train",
    type=int,
    default=100,
    help="percentage of class1 samples in training set",
)
parser.add_argument("--hot_enc", type=int, default=0, help="hot enc set")
parser.add_argument("--num_epochs", type=int, default=100, help="num epochs")
parser.add_argument(
    "--resnet_type", type=str, default="resnet10", help="type of renset to be used"
)
# Objective function parameters
parser.add_argument(
    "--training_type",
    type=str,
    help="Loss function type. \
                    Options: 'ASYM_LM', 'SYM_LM', 'ASYM_FL', 'SYM_FL', 'WBCE', \
                    'cb_BCE', 'MBAUC', otherwise 'BCE'",
)
parser.add_argument("--gamma", help="set gamma hyperparam for FL")
parser.add_argument("--m", type=float, help="set m hyperparam for LM")
parser.add_argument("--c", type=float, help="set c hyperparam for WBCE")
parser.add_argument("--beta", type=float, help="set c hyperparam for WBCE")
# ALM parameters
parser.add_argument("--use_ALM", type=int, default=0, help="use ALM")
parser.add_argument(
    "--introd_ALM_from", type=int, default=0, help="introduce ALM at this epoch"
)
parser.add_argument("--p2_norm", type=float, default=1, help="2p norm for ALM")
parser.add_argument("--mu", type=float, default=0.1, help="set initial mu")
parser.add_argument("--mu_limit", type=float, default=0.06, help="maximum mu allowed")
parser.add_argument(
    "--rho", type=int, default=2, help="coefficient in ALM for squared penalty term"
)
parser.add_argument("--delta", type=float, default=0, help="margin to improve ALM")
parser.add_argument("--cap_mu", type=int, default=1, help="set capping of mu")
parser.add_argument(
    "--hnm_mode",
    type=str,
    default="all",
    help="ALM mining mode: 'all', 'random', 'topk', 'semi_hard', or 'semi_hard_topk'",
)
parser.add_argument(
    "--hnm_topk",
    type=int,
    default=10,
    help="number of hard negatives selected per positive sample when top-k mining is active",
)
parser.add_argument(
    "--hnm_warmup_epochs",
    type=int,
    default=5,
    help="number of warmup epochs before hard negative mining starts",
)
parser.add_argument(
    "--hnm_random_k",
    type=int,
    default=0,
    help="number of random negatives mixed with hard negatives for stability",
)
parser.add_argument(
    "--hnm_semi_hard_upper",
    type=float,
    default=0.5,
    help="upper bound on violation values used to define semi-hard negatives",
)
parser.add_argument(
    "--alm_q_clip",
    type=float,
    default=0.0,
    help="maximum aggregate ALM violation; non-positive disables clipping",
)
parser.add_argument(
    "--alm_lambda_clip",
    type=float,
    default=0.0,
    help="absolute ALM multiplier cap; non-positive disables clipping",
)
parser.add_argument(
    "--grad_clip_norm",
    type=float,
    default=0.0,
    help="maximum gradient norm; non-positive disables clipping",
)
parser.add_argument(
    "--weight_dec", type=float, default=0, help="weight decay of optimizer"
)

# Load JSON values as parser defaults so explicitly supplied command-line
# options remain authoritative for reproducible ablations and sweeps.
initial_args, _ = parser.parse_known_args()
config_defaults = {}
if initial_args.name_file_bs == "":
    parser.error("--name_file_bs is required")
with open(os.path.join("configs/baseline/", initial_args.name_file_bs + ".json")) as f:
    config_defaults.update(json.load(f))
print("JSON file baseline:", initial_args.name_file_bs)

if initial_args.name_file_alm != "":
    with open(os.path.join("configs/ALM/", initial_args.name_file_alm + ".json")) as f:
        config_defaults.update(json.load(f))
    print("JSON file ALM:", initial_args.name_file_alm)

parser.set_defaults(**config_defaults)
args = vars(parser.parse_args())
args["seed_list"] = parse_seed_list(args["seed_list"])
if not args["seed_list"]:
    parser.error("--seed_list must contain at least one integer seed")

if args["name_file_alm"] != "":
    if args["mu"] < 1e-3:
        args["mu_limit"] = args["mu"] * args["rho"] ** 5
    else:
        args["mu_limit"] = args["mu"] * args["rho"] ** 3

if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(args["gpu_id"]))
else:
    device = torch.device("cpu")
print("Running on device: ", device)


########################################
# Initialize all the useful variables
########################################
# Dataset settings
path_to_dataset = "datasets/" + args["dataset_name"] + "/"
print(
    "You have selected dataset {} and network {}. ".format(
        args["dataset_name"], args["resnet_type"]
    )
)
if args["dataset_name"] not in ("cifar10", "cifar100", "svhn"):
    parser.error("Unsupported dataset_name: {}".format(args["dataset_name"]))
if args["dataset_name"] == "svhn":
    normalize_mean = (0.4377, 0.4438, 0.4728)
    normalize_std = (0.1980, 0.2010, 0.1970)
else:
    normalize_mean = (0.4914, 0.4822, 0.4465)
    normalize_std = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ]
)
transform_val_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ]
)
if args["dataset_name"] == "cifar100":
    new_majority_labels = np.array(
        [72, 4, 95, 30, 55]
    )  # Superclass fishes - to reproduce the paper
    new_minority_labels = np.array([73])  # Subclass sharks - to reproduce the paper
    val_dataset = BinaryBalancedCIFAR100(
        root=path_to_dataset,
        tot_classes=2,
        first_class_labels=new_majority_labels,
        second_class_labels=new_minority_labels,
        val_samples_per_cls=50,
        train=True,
        download=bool(args["download"]),
        transform=transform_val_test,
    )
    test_dataset = BinaryBalancedCIFAR100(
        root=path_to_dataset,
        tot_classes=2,
        first_class_labels=new_majority_labels,
        second_class_labels=new_minority_labels,
        val_samples_per_cls=100,
        train=False,
        download=bool(args["download"]),
        transform=transform_val_test,
    )
elif args["dataset_name"] == "cifar10":
    # Five majority classes versus one minority class; four classes are unused.
    new_majority_labels = np.array([0, 2, 3, 4, 5])
    new_minority_labels = np.array([1])
    val_dataset = BinaryBalancedCIFAR10(
        root=path_to_dataset,
        tot_classes=2,
        first_class_labels=new_majority_labels,
        second_class_labels=new_minority_labels,
        val_samples_per_cls=50,
        train=True,
        download=bool(args["download"]),
        transform=transform_val_test,
    )
    test_dataset = BinaryBalancedCIFAR10(
        root=path_to_dataset,
        tot_classes=2,
        first_class_labels=new_majority_labels,
        second_class_labels=new_minority_labels,
        val_samples_per_cls=100,
        train=False,
        download=bool(args["download"]),
        transform=transform_val_test,
    )
elif args["dataset_name"] == "svhn":
    # Five majority digits versus digit 1 as the minority class.
    new_majority_labels = np.array([2, 3, 4, 5, 6])
    new_minority_labels = np.array([1])
    val_dataset = BinaryBalancedSVHN(
        root=path_to_dataset,
        tot_classes=2,
        first_class_labels=new_majority_labels,
        second_class_labels=new_minority_labels,
        val_samples_per_cls=50,
        train=True,
        download=bool(args["download"]),
        transform=transform_val_test,
    )
    test_dataset = BinaryBalancedSVHN(
        root=path_to_dataset,
        tot_classes=2,
        first_class_labels=new_majority_labels,
        second_class_labels=new_minority_labels,
        val_samples_per_cls=100,
        train=False,
        download=bool(args["download"]),
        transform=transform_val_test,
    )
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

# Initialize parameters
seeds = args["seed_list"]
max_validation_auc_runs = np.zeros(len(seeds))
val_matrix_logits = np.zeros((len(val_loader.dataset), len(seeds)))
test_matrix_logits = np.zeros((len(test_loader.dataset), len(seeds)))
test_auc_array = np.zeros(len(seeds))
test_auprc_array = np.zeros(len(seeds))
test_fpr90_array = np.zeros(len(seeds))
test_fpr95_array = np.zeros(len(seeds))
test_fpr98_array = np.zeros(len(seeds))
num_classes = 2

MODEL_NAME = args["model_name"]
print(
    "Model name {}, ratio classes {} training, epochs {}, LR decrease {}.".format(
        MODEL_NAME, args["ratio_pos_train"], args["num_epochs"], args["lr_decrease"]
    )
)

for seed_idx, seed in enumerate(seeds):
    print("Training model: {}. Run num. {}.".format(MODEL_NAME, seed))
    print("Training type", args["training_type"])
    if args["use_ALM"] == 1:
        print("ALM selected.")
        print(
            "ALM params: rho={}, mu={}, delta={}".format(
                args["rho"], args["mu"], args["delta"]
            )
        )
        print(
            "HNM params: mode={}, topk={}, warmup={}, random_k={}, semi_hard_upper={}".format(
                args["hnm_mode"],
                args["hnm_topk"],
                args["hnm_warmup_epochs"],
                args["hnm_random_k"],
                args["hnm_semi_hard_upper"],
            )
        )
        print(
            "ALM stability: q_clip={}, lambda_clip={}, grad_clip_norm={}".format(
                args["alm_q_clip"],
                args["alm_lambda_clip"],
                args["grad_clip_norm"],
            )
        )
    else:
        print("ALM not selected.")
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    SEED_STR = "seed" + str(seed)
    MODEL_DIR = os.path.join(
        "checkpoint/", args["subfolder_name"], MODEL_NAME, SEED_STR
    )
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if args["dataset_name"] == "cifar100":
        train_dataset = BinaryUnbalancedCIFAR100(
            root=path_to_dataset,
            imb_factor=1 / args["ratio_pos_train"],
            seed=seed,
            new_majority_labels=new_majority_labels,
            new_minority_labels=new_minority_labels,
            val_samples_per_cls=50,
            perc_per_run=0.9,
            train=True,
            download=bool(args["download"]),
            transform=transform_train,
        )
        lr = 0.001
    elif args["dataset_name"] == "cifar10":
        train_dataset = BinaryUnbalancedCIFAR10(
            root=path_to_dataset,
            imb_factor=1 / args["ratio_pos_train"],
            seed=seed,
            new_majority_labels=new_majority_labels,
            new_minority_labels=new_minority_labels,
            val_samples_per_cls=50,
            perc_per_run=0.9,
            train=True,
            download=bool(args["download"]),
            transform=transform_train,
        )
        lr = 0.001
    elif args["dataset_name"] == "svhn":
        train_dataset = BinaryUnbalancedSVHN(
            root=path_to_dataset,
            imb_factor=1 / args["ratio_pos_train"],
            seed=seed,
            new_majority_labels=new_majority_labels,
            new_minority_labels=new_minority_labels,
            val_samples_per_cls=50,
            perc_per_run=0.9,
            train=True,
            download=bool(args["download"]),
            transform=transform_train,
        )
        lr = 0.001

    per_class_samples = train_dataset.get_tot_classes_list(tot_classes=2)
    print(
        "Training class counts: majority={}, minority={}, actual_ratio=1:{:.3f}".format(
            per_class_samples[0],
            per_class_samples[1],
            per_class_samples[0] / per_class_samples[1],
        )
    )
    num_maj = len(train_dataset.targets) - sum(
        train_dataset.targets
    )  # class 0 majority
    use_norm = True if args["training_type"] == "LDAM" else False
    if args["resnet_type"].lower() == "resnet10":
        net = ResNet10(num_classes, use_norm=use_norm)
    elif args["resnet_type"].lower() == "resnet18":
        net = ResNet18(num_classes, use_norm=use_norm)
    else:
        parser.error("Unsupported resnet_type: {}".format(args["resnet_type"]))
    net.to(device)

    #     ######################################
    #     #Train, Validate, Test
    #     ######################################
    if args["train"] == 1:
        max_validation_auc_runs[seed_idx], val_matrix_logits[:, seed_idx] = train(
            net,
            args["num_epochs"],
            args["subfolder_name"],
            MODEL_NAME,
            MODEL_DIR,
            seed,
            per_class_samples,
            args["bsize"],
            args["num_workers"],
            lr,
            args["lr_decrease"],
            args["patience_lr"],
            args["patience_early_stopping"],
            num_maj,
            args["ratio_pos_train"],
            args["training_type"],
            args["gamma"],
            args["m"],
            args["c"],
            args["beta"],
            args["continue_train"],
            args["continue_train_epochs"],
            args["baseline_cont_train"],
            args["use_ALM"],
            args["introd_ALM_from"],
            args["p2_norm"],
            train_dataset,
            val_loader,
            args["rho"],
            args["mu"],
            args["mu_limit"],
            args["cap_mu"],
            args["delta"],
            args["hnm_mode"],
            args["hnm_topk"],
            args["hnm_warmup_epochs"],
            args["hnm_random_k"],
            args["hnm_semi_hard_upper"],
            args["alm_q_clip"],
            args["alm_lambda_clip"],
            args["grad_clip_norm"],
            device,
        )
    else:
        validation_history_files = [
            os.path.join(MODEL_DIR, MODEL_NAME + "_loss_AUC_values.csv"),
            os.path.join(MODEL_DIR, MODEL_NAME + "_val_loss_AUC_values.csv"),
        ]
        validation_history_file = next(
            (path for path in validation_history_files if os.path.exists(path)), None
        )
        if validation_history_file is not None:
            validation_history = pd.read_csv(validation_history_file)
            max_validation_auc_runs[seed_idx] = validation_history[
                "Val AUC"
            ].max()
        else:
            warnings.warn(
                "Validation history is missing for seed {}: {}".format(
                    seed, ", ".join(validation_history_files)
                )
            )

    test_auc_array[seed_idx] = test(
        net,
        test_loader,
        MODEL_DIR,
        MODEL_NAME,
        args["training_type"],
        device,
    )

    # To keep track of the results on test set either unbalanced in the same way as training or balanced
    df_fv = pd.read_csv(os.path.join(MODEL_DIR, MODEL_NAME + "_LOGITS_and_GT_fv.csv"))
    test_labels = df_fv["GT"].to_numpy()
    logits = df_fv["Z"].to_numpy()
    test_matrix_logits[:, seed_idx] = logits
    test_auprc_array[seed_idx] = average_precision_score(test_labels, logits)
    test_fpr90_array[seed_idx] = fpr_at_tpr(test_labels, logits, 0.90)
    test_fpr95_array[seed_idx] = fpr_at_tpr(test_labels, logits, 0.95)
    test_fpr98_array[seed_idx] = fpr_at_tpr(test_labels, logits, 0.98)


# Persist per-seed metrics and the exact resolved configuration for statistical
# analysis. The ensemble result remains a separate, secondary report.
experiment_root = os.path.join("checkpoint", args["subfolder_name"], MODEL_NAME)
os.makedirs(experiment_root, exist_ok=True)
run_mode = "train" if args["train"] == 1 else "eval"
seed_range_tag = "seeds{}-{}".format(min(seeds), max(seeds))
pd.DataFrame(
    {
        "seed": seeds,
        "validation_auc_max": max_validation_auc_runs,
        "test_roc_auc": test_auc_array,
        "test_auprc": test_auprc_array,
        "test_fpr_at_tpr90": test_fpr90_array,
        "test_fpr_at_tpr95": test_fpr95_array,
        "test_fpr_at_tpr98": test_fpr98_array,
    }
).to_csv(
    os.path.join(
        experiment_root,
        "{}_{}_seed_metrics_{}.csv".format(MODEL_NAME, run_mode, seed_range_tag),
    ),
    index=False,
)

average_test_logits = np.average(test_matrix_logits, axis=1)
pd.DataFrame(
    {
        "seed_count": [len(seeds)],
        "ensemble_roc_auc": [roc_auc_score(test_labels, average_test_logits)],
        "ensemble_auprc": [
            average_precision_score(test_labels, average_test_logits)
        ],
        "ensemble_fpr_at_tpr90": [
            fpr_at_tpr(test_labels, average_test_logits, 0.90)
        ],
        "ensemble_fpr_at_tpr95": [
            fpr_at_tpr(test_labels, average_test_logits, 0.95)
        ],
        "ensemble_fpr_at_tpr98": [
            fpr_at_tpr(test_labels, average_test_logits, 0.98)
        ],
    }
).to_csv(
    os.path.join(
        experiment_root,
        "{}_{}_ensemble_metrics_{}.csv".format(
            MODEL_NAME, run_mode, seed_range_tag
        ),
    ),
    index=False,
)

config_snapshot = dict(args)
config_snapshot["device"] = str(device)
config_snapshot["resolved_experiment_root"] = experiment_root
with open(
    os.path.join(
        experiment_root,
        "{}_{}_config_snapshot_{}.json".format(MODEL_NAME, run_mode, seed_range_tag),
    ),
    "w",
) as f:
    json.dump(config_snapshot, f, indent=2, sort_keys=True)

# Calculate and print final results
if args["train"] == 1:
    validation_ensembling(val_matrix_logits, val_loader.dataset.targets)

final_ensembling_results(
    args["subfolder_name"],
    MODEL_DIR,
    MODEL_NAME,
    test_matrix_logits,
    test_auc_array,
    max_validation_auc_runs,
)
