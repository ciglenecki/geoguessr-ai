import argparse
import os
from pathlib import Path
import sys
from typing import List
import pandas as pd
import numpy as np
import shutil
from distutils.dir_util import copy_tree
from config import DEFAULT_SPACING
import preprocess_csv_concat
from utils_paths import PATH_DATA_SUBSET_EXTERNAL, PATH_DATA_SUBSET_ORIGINAL
import preprocess_csv_create_rich_static


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dirs",
        metavar="dir",
        nargs="+",
        type=str,
        help="Dataset root directories that will be transformed into a single dataset",
        default=[PATH_DATA_SUBSET_EXTERNAL, PATH_DATA_SUBSET_ORIGINAL],
    )
    parser.add_argument(
        "--out-dir",
        metavar="dir",
        type=str,
        help="Directory where compelte dataset will be placed",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images from dataset directories to the new directory. You don't need to do this as later on you will be able to pass multiple dataset directories to various scripts.",
        default=False,
    )
    parser.add_argument(
        "--spacing",
        type=float,
        help="""Spacing that will be used to create a grid of polygons. Different spacings produce different number of classes
        0.7 spacing => ~31 classes
        0.5 spacing => ~55 classes
        0.4 spacing => ~75 classes
        0.3 spacing => ~115 classes
        """,
        default=DEFAULT_SPACING,
    )

    return parser.parse_args(args)


def copy_images(dataset_dirs: List[Path], out_dir: Path):
    images_out_path = Path(out_dir, "images")
    os.makedirs(images_out_path, exist_ok=True)
    for dataset_dir in dataset_dirs:
        copy_tree(str(Path(dataset_dir, "images")), str(Path(out_dir, "images")))
    print("Images saved to: '{}'".format(images_out_path))


def save_new_csv(dataset_dirs: List[Path], out_dir: Path) -> str:
    csv_paths = [str(Path(dataset_dir, "data.csv")) for dataset_dir in dataset_dirs]

    path_new_csv = str(Path(out_dir, "data.csv"))
    preprocess_csv_concat.main(
        [
            "--csv",
            *csv_paths,
            "--out",
            path_new_csv,
        ]
    )
    return path_new_csv


def concat_datasets(dataset_dirs: List[Path], out_dir: Path, should_copy_images: bool):
    if should_copy_images:
        copy_images(dataset_dirs, out_dir)
    save_new_csv(dataset_dirs, out_dir)
    return out_dir


def main(args):
    args = parse_args(args)
    dataset_dirs = args.dataset_dirs
    should_copy_images = args.copy_images
    if args.out_dir:
        out_dir = args.out_dir
    else:
        parent_dir = dataset_dirs[0].parent
        out_dir = Path(parent_dir, "complete_subset")

    out_dir = concat_datasets(dataset_dirs, out_dir, should_copy_images=False)
    path_csv_out = preprocess_csv_create_rich_static.main(
        [
            "--csv",
            str(Path(out_dir, "data.csv")),
            "--spacing",
            str(args.spacing),
        ]
    )


if __name__ == "__main__":
    main(sys.argv[1:])
