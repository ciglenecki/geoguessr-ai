import argparse
import sys
from glob import glob
from os.path import exists
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from utils_dataset import DatasetSplitType
from utils_functions import flatten, get_dirs_only


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dirs",
        type=str,
        help="Image directories (directories with UUIDs) for which the norm and std will be caculated.",
        nargs="+",
        required=True,
    )
    return parser.parse_args(args)


def calculate_norm_std(image_dirs: list[Path]):
    """
    Caculates the mean and std of the multiple train dataset directories
    Args:
        image_dirs: list of dataset directory paths which the mean and std will be caculated for.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    degrees = ["0", "90", "180", "270"]
    transform = transforms.ToTensor()

    for image_dir in tqdm(image_dirs):

        uuid_dir_paths = get_dirs_only(Path(image_dir))
        uuids = [Path(uuid_dir_path).stem for uuid_dir_path in uuid_dir_paths]

        for uuid in tqdm(uuids):
            location_dir = Path(image_dir, uuid)
            image_filepaths = [
                Path(location_dir, "{}.jpg".format(degree)) for degree in degrees
            ]
            images = [
                transform(Image.open(image_path)) for image_path in image_filepaths
            ]
            for image in images:
                channels_sum += torch.mean(image, dim=[1, 2]).detach()
                channels_squared_sum += torch.mean(image**2, dim=[1, 2]).detach()
                num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    print("Datasets:", image_dirs)
    print("Train dataset mean: " + str(mean))
    print("Train dataset std: " + str(std))
    return mean, std


def main(args):
    args = parse_args(args)
    calculate_norm_std(args.image_dirs)


if __name__ == "__main__":
    main(sys.argv[1:])
