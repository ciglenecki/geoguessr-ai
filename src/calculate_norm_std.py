import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import transforms

from defaults import DEFAULT_LOAD_DATASET_IN_RAM
from utils_dataset import DatasetSplitType


def calculate_norm_std(dataset_dir):
    path_images = Path(dataset_dir, DatasetSplitType.TRAIN.value)
    uuids = sorted(next(os.walk(path_images))[1])
    degrees = ["0", "90", "180", "270"]
    load_dataset_in_ram = DEFAULT_LOAD_DATASET_IN_RAM
    transform = transforms.ToTensor()
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    image_cache = {}
    for uuid in uuids:
        image_dir = Path(path_images, uuid)
        image_filepaths = [Path(image_dir, "{}.jpg".format(degree)) for degree in degrees]
        cache_item = [Image.open(image_path) for image_path in
                      image_filepaths] if load_dataset_in_ram else image_filepaths
        image_cache[uuid] = cache_item

    for images in image_cache.values():
        images = [transform(Image.open(image_path)) for image_path in images]
        for image in images:
            channels_sum += torch.mean(image, dim=[1, 2])
            channels_squared_sum += torch.mean(image ** 2, dim=[1, 2])
            num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print("Dataset mean: " + str(mean))
    print("Dataset std: " + str(std))

    return mean, std

