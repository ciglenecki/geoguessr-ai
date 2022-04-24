
from glob import glob
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision.transforms import transforms

from utils_functions import flatten
from utils_dataset import DatasetSplitType


def calculate_norm_std(dataset_dirs: List[Path]):
    """
    Caculates the mean and std of the multiple train dataset directories 
    Args:
        dataset_dirs: list of dataset directory paths which the mean and std will be caculated for.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    degrees = ["0", "90", "180", "270"]
    transform = transforms.ToTensor()
    
    for dataset_dir in dataset_dirs:
        path_images = Path(dataset_dir, "images", DatasetSplitType.TRAIN.value)
        uuid_dir_paths = glob(str(Path(path_images, "*")))
        uuids = [Path(uuid_dir_path).stem for uuid_dir_path in uuid_dir_paths]

        for uuid in uuids:
            image_dir = Path(path_images, uuid)
            image_filepaths = [Path(image_dir, "{}.jpg".format(degree)) for degree in degrees]
            images = [transform(Image.open(image_path)) for image_path in image_filepaths]
            for image in images:
                channels_sum += torch.mean(image, dim=[1, 2]).detach()
                channels_squared_sum += torch.mean(image**2, dim=[1, 2]).detach()
                num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    print("Train dataset mean: " + str(mean))
    print("Train dataset std: " + str(std))
    return mean, std
