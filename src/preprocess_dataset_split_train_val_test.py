"""
Unpack the original dataset images directory (`data`) to train/val/test directories.
By default, directories will be created on the same level of the original `data` directory.

Reproducability is achieved by using the `sorted` function once uuids are listed.
"""

import argparse
import distutils.dir_util
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import DEFAULT_TEST_FRAC, DEFAULT_TRAIN_FRAC, DEFAULT_VAL_FRAC
from utils_functions import is_valid_dir, is_valid_fractions_array, split_by_ratio


def parse_args(args):
    print(args)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=is_valid_dir,
        help="Path to the directory that contains UUID subdirectories.",
        required=True,
    )
    parser.add_argument(
        "--out",
        metavar="dir",
        help="Directory where train/val/test directories will be created.",
    )

    parser.add_argument(
        "--split-ratios",
        metavar="[float, float, float]",
        nargs=3,
        default=[DEFAULT_TRAIN_FRAC, DEFAULT_VAL_FRAC, DEFAULT_TEST_FRAC],
        type=float,
        help="Fractions of train, validation and test that will be used to split the dataset. E.g. 0.8 0.1 0.1",
    )

    args = parser.parse_args(args)
    return args


def create_train_val_test_dirs(out_dir: Path):
    for x in ["train", "val", "test"]:
        Path(out_dir, x).mkdir(parents=True, exist_ok=True)


def main(args):
    args = parse_args(args)
    print(args)

    image_dir = args.image_dir
    out_dir = args.image_dir if args.out is None else args.out
    split_ratios = args.split_ratios

    if sum(split_ratios) != 1:
        raise argparse.ArgumentError(split_ratios, "There has to be 3 fractions (train, val, test) that sum to 1")

    uuids = sorted(next(os.walk(image_dir))[1])
    uuid_indices = np.arange(len(uuids))

    train_frac, val_frac, test_frac = split_ratios
    train_indices, val_indices, test_indices = split_by_ratio(
        uuid_indices, train_frac, val_frac, test_frac, use_whole_array=True
    )

    create_train_val_test_dirs(out_dir)

    list_of_indices = [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]
    for split_name, indices in list_of_indices:
        for i in tqdm(
            range(indices[0], indices[-1] + 1),
            desc="Copying {} images".format(split_name),
        ):
            uuid = uuids[i]
            path_source_dir = Path(image_dir, uuid)
            path_dest_dir = Path(out_dir, split_name, uuid)
            distutils.dir_util.copy_tree(str(path_source_dir), str(path_dest_dir))


if __name__ == "__main__":
    main(sys.argv[1:])
