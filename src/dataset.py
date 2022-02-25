from __future__ import annotations, print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from pathlib import Path
from utils_paths import PATH_DATA_RAW
import pandas as pd


class GeoguesserDatasetRaw(Dataset):
    def __init__(self, root_dir: Path = PATH_DATA_RAW) -> None:
        self.path_images = Path(root_dir, "data")
        self.path_csv = Path(root_dir, "data.csv")
        self.df_csv = pd.read_csv(self.path_csv, index_col="uuid")
        self.uuids = sorted(os.listdir(self.path_images))
        self.degrees = ["0", "90", "180", "270"]

    def name_without_extension(self, filename: Path | str):
        return Path(filename).stem

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        """
        Gets the uuid
        Loads images via the uuid
        Loads latitude and longitude via the csv and uuid
        """
        uuid = self.uuids[index]
        image_dir = Path(self.path_images, uuid)
        image_filepaths = list(map(lambda degree: Path(image_dir, "{}.jpg".format(degree)), self.degrees))
        images = list(map(lambda x: Image.open(x), image_filepaths))
        lat_lng = self.df_csv.loc[uuid, :]
        latitude, longitude = float(lat_lng["latitude"]), float(lat_lng["longitude"])
        return images, latitude, longitude
