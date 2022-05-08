""" Similar to caculate_norm_std.py but uses dataloader instead. A little bit faster. """
from __future__ import annotations, division, print_function

from glob import glob
from pathlib import Path
from typing import List

import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from config import DEFAULT_IMAGE_SIZE
from utils_dataset import DatasetSplitType
from utils_functions import flatten
from utils_paths import PATH_DATA, PATH_DATA_EXTERNAL, PATH_DATA_ORIGINAL


class DatasetFlat(Dataset):
    def __init__(
        self,
        dataset_dirs: List[Path],
        dataset_type: DatasetSplitType,
        image_transform=transforms.Compose(
            [transforms.Resize(DEFAULT_IMAGE_SIZE), transforms.ToTensor()],
        ),
    ) -> None:
        self.image_paths = flatten(
            [
                glob(str(Path(dataset_dir, dataset_type.value, "**/*.jpg")), recursive=True)
                for dataset_dir in dataset_dirs
            ]
        )
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index])
        return self.image_transform(image)


if __name__ == "__main__":
    dataset_dirs = [PATH_DATA_ORIGINAL, PATH_DATA_EXTERNAL, Path(PATH_DATA, "external2")]
    dataset = DatasetFlat(
        dataset_dirs,
        DatasetSplitType.TRAIN,
    )

    def get_mean_and_std(dataloader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data in tqdm(dataloader):
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches

        std = (channels_squared_sum / num_batches - mean**2) ** 0.5
        print(mean, std)
        return mean, std

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)
    mean, std = get_mean_and_std(loader)
