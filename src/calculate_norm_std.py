import os
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import transforms
from utils_dataset import DatasetSplitType


def calculate_norm_std(dataset_dirs):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for dataset_dir in dataset_dirs:
        path_images = Path(dataset_dir, "images", DatasetSplitType.TRAIN.value)
        uuids = sorted(next(os.walk(path_images))[1])
        degrees = ["0", "90", "180", "270"]
        transform = transforms.ToTensor()

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
