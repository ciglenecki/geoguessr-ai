"""
Input arguments for the train.py file. To see the list of all arguments call `pyhton3 src/train.py -h`
"""
from __future__ import annotations, division, print_function

import argparse
from typing import Dict, Tuple

import pytorch_lightning as pl
from torchvision.models.resnet import model_urls as resnet_model_urls

from config import (
    DEAFULT_NUM_WORKERS,
    DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_FRAC,
    DEFAULT_EPOCHS,
    DEFAULT_FINETUNING_EPOCH_PERIOD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LR,
    DEFAULT_LR_FINETUNE,
    DEFAULT_MODEL,
    DEFAULT_PRETRAINED,
    DEFAULT_SCHEDULER,
    DEFAULT_TEST_FRAC,
    DEFAULT_TRAIN_FRAC,
    DEFAULT_UNFREEZE_LAYERS_NUM,
    DEFAULT_VAL_FRAC,
    DEFAULT_WEIGHT_DECAY,
    LOG_EVERY_N,
)
from config import DEFAULT_OPTIMIZER
from utils_functions import (
    is_between_0_1,
    is_positive_int,
    is_valid_dir,
    is_valid_fractions_array,
    is_valid_image_size,
    is_valid_unfreeze_arg,
)
from utils_paths import PATH_DATA_EXTERNAL, PATH_DATA_RAW, PATH_REPORT
from utils_train import OptimizerType, SchedulerType

ARGS_GROUP_NAME = "General arguments"


def parse_args_train() -> Tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser()

    lightning_parser = pl.Trainer.add_argparse_args(parser)
    lightning_parser.set_defaults(log_every_n_steps=LOG_EVERY_N)

    user_group = parser.add_argument_group(ARGS_GROUP_NAME)
    user_group.add_argument(
        "--split-ratios",
        metavar="[float, float, float]",
        nargs=3,
        default=[DEFAULT_TRAIN_FRAC, DEFAULT_VAL_FRAC, DEFAULT_TEST_FRAC],
        type=is_valid_fractions_array,
        help="Fractions of train, validation and test that will be used to split the dataset",
    )
    user_group.add_argument(
        "--regression",
        action="store_true",
        default=False,
        help="Select regression model for training",
    )
    user_group.add_argument(
        "--dataset-frac",
        metavar="float",
        default=DEFAULT_DATASET_FRAC,
        type=is_between_0_1,
        help="Size of the dataset that will be trained",
    )
    user_group.add_argument(
        "--image-size",
        metavar="int",
        default=DEFAULT_IMAGE_SIZE,
        type=is_valid_image_size,
        help="Image size",
    )
    user_group.add_argument(
        "--num-workers",
        metavar="int",
        default=DEAFULT_NUM_WORKERS,
        type=is_positive_int,
        help="Number of workers",
    )
    user_group.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        type=str,
        help="Models used for training",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2",
        ],
    )
    user_group.add_argument(
        "-l",
        "--lr",
        default=DEFAULT_LR,
        type=float,
        help="Learning rate",
    )
    user_group.add_argument(
        "--dataset-dirs",
        metavar="dir",
        nargs="+",
        type=is_valid_dir,
        help="Dataset root directories that will be used for training",
        default=[PATH_DATA_RAW, PATH_DATA_EXTERNAL],
    )

    user_group.add_argument(
        "--cached-df",
        type=str,
        help="e.g. ata/raw.ignore/data__num_class_259__spacing_0.2.csv => Filepath to cached dataframe",
    )

    user_group.add_argument(
        "--output-report",
        metavar="dir",
        type=str,
        help="Directory where report file will be created.",
        default=PATH_REPORT,
    )
    user_group.add_argument(
        "-b",
        "--unfreeze-blocks",
        metavar="['all', 'layer3.2', n]",
        type=is_valid_unfreeze_arg,
        help="Number of trainable blocks. Parameters of trainable block will be updated (required_grad=True) during the training. This argument changes nothing if argument `--pretrained` isn't set. If model isn't pretrained its weights are random.It doesn't make sense to freeze blocks which have random (untrained) weights.",
        default=DEFAULT_UNFREEZE_LAYERS_NUM,
    )
    user_group.add_argument(
        "-p",
        "--pretrained",
        type=bool,
        help="Load pretrained model.",
        default=DEFAULT_PRETRAINED,
    )
    user_group.add_argument(
        "--shuffle-before-splitting",
        type=bool,
        help="The dataset will be shuffled before splitting dataset to train/val/test",
        default=DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING,
    )

    user_group.add_argument(
        "--recaculate-normalization",
        help="Calls calculate_norm_std during before training to ensure that images are properly normalize with training data only.",
        action="store_true",
        default=False,
    )

    user_group.add_argument(
        "-q",
        "--quick",
        help="Simulates --limit_train_batches 2 --limit_val_batches 2 --limit_test_batches 2 --image-size 28",
        action="store_true",
        default=False,
    )

    user_group.add_argument(
        "-t",
        "--trainer-checkpoint",
        help=".ckpt file, automatically restores model, epoch, step, LR schedulers, etc...",
        metavar="path",
        type=str,
    )

    user_group.add_argument(
        "-u",
        "--unfreeze-at-epoch",
        help="Backbone Finetuning. Trains only last layer for n epoches.",
        metavar="N",
        type=is_positive_int,
        default=DEFAULT_FINETUNING_EPOCH_PERIOD,
    )

    user_group.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    user_group.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    user_group.add_argument(
        "--lr-finetune",
        type=float,
        default=DEFAULT_LR_FINETUNE,
    )

    user_group.add_argument(
        "--use-single-images",
        action="store_true",
        help="Use single image as an input to the model",
    )

    user_group.add_argument(
        "--scheduler",
        default=DEFAULT_SCHEDULER,
        type=str,
        choices=[scheduler_type.value for scheduler_type in SchedulerType],
    )

    user_group.add_argument(
        "--optimizer",
        default=DEFAULT_OPTIMIZER,
        type=str,
        choices=[optimizer_type.value for optimizer_type in OptimizerType],
    )

    user_group.add_argument(
        "--epochs",
        default=DEFAULT_EPOCHS,
        type=is_positive_int,
    )

    args = parser.parse_args()

    """Separate Namespace into two Namespaces"""
    args_dict: Dict[str, argparse.Namespace] = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if group.title:
            args_dict[group.title] = argparse.Namespace(**group_dict)

    args, pl_args = args_dict[ARGS_GROUP_NAME], args_dict["pl.Trainer"]

    """User arguments that override PyTorch Lightning arguments"""
    if args.quick:
        pl_args.limit_train_batches = 4
        pl_args.limit_val_batches = 4
        pl_args.limit_test_batches = 4
        pl_args.log_every_n_steps = 1
        args.image_size = 16
        args.batch_size = 2
    return args, pl_args


if __name__ == "__main__":
    pass
