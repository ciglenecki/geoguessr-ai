from __future__ import annotations, division, print_function

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils_paths import PATH_DATA_RAW


class GeoguesserDatasetRaw(Dataset):
    """
    Pytorch Dataset class which defines how elements are fetched from the soruce (directory)

    1. Holds pointers to the data (images and coordinates)
    2. Fetches them lazly when __getitem__ is called
    """

    def __init__(
        self,
        root_dir: Path = PATH_DATA_RAW,
        image_transform: None | transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        coordinate_transform: None | Callable = lambda x, y: np.array([x, y]).astype("float"),
    ) -> None:
        self.image_transform = image_transform
        self.coordinate_transform = coordinate_transform
        self.path_images = Path(root_dir, "data")
        self.path_csv = Path(root_dir, "data.csv")
        self.df_csv = pd.read_csv(self.path_csv, index_col="uuid")
        self.degrees = ["0"]
        """
        os.walk is a generator and calling next will get the first result in the form of a 3-tuple (dirpath, dirnames, filenames). The [1] index returns only the dirnames from that tuple.
        """
        self.uuids = sorted(next(os.walk(self.path_images))[1])

    def name_without_extension(self, filename: Path | str):
        return Path(filename).stem

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, index: int):
        """
        Gets the uuid
        Loads images via the uuid
        Loads latitude and longitude via the csv and uuid
        Applies transforms
        """
        uuid = self.uuids[index]
        image_dir = Path(self.path_images, uuid)
        image_filepaths = list(map(lambda degree: Path(image_dir, "{}.jpg".format(degree)), self.degrees))
        images = list(map(lambda x: Image.open(x), image_filepaths))
        lat_lng = self.df_csv.loc[uuid, :]
        latitude, longitude = float(lat_lng["latitude"]), float(lat_lng["longitude"])

        if self.image_transform is not None:
            transform = self.image_transform
            images = list(map(lambda i: transform(i), images))
        if self.coordinate_transform is not None:
            transform = self.coordinate_transform
            latitude, longitude = self.coordinate_transform(latitude, longitude)

        return images[0], latitude, longitude


if __name__ == "__main__":
    print("This file shouldn't be called as a script unless used for debugging.")
    dataset = GeoguesserDatasetRaw()
    print(dataset.__getitem__(2))
