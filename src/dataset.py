from __future__ import annotations, division, print_function

import os
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils_env import DEFAULT_LOAD_DATASET_IN_RAM

from utils_functions import one_hot_encode
from utils_paths import PATH_DATA_RAW
import coords_decorate_csv


class GeoguesserDataset(Dataset):
    """
    Pytorch Dataset class which defines how elements are fetched from the soruce (directory)

    1. Holds pointers to the data (images and coordinates)
    2. Fetches them lazly when __getitem__ is called
    """

    def __init__(
        self,
        dataset_dir: Path = PATH_DATA_RAW,
        image_transform: None | transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        coordinate_transform: None | Callable = lambda x, y: np.array([x, y]).astype("float"),
        cached_df=None,
        load_dataset_in_ram=DEFAULT_LOAD_DATASET_IN_RAM,
    ) -> None:
        print("GeoguesserDataset init")
        super().__init__()
        self.degrees = ["0", "90", "180", "270"]
        self.image_transform = image_transform
        self.coordinate_transform = coordinate_transform
        self.path_images = Path(dataset_dir, "data")
        self.df_csv = pd.read_csv(Path(cached_df)) if cached_df else coords_decorate_csv.main(["--spacing", str(0.2), "--no-out"])

        """ Filter the dataframe, only include rows for images that exist and remove polygons with no data"""
        self.df_csv = self.filter_df_rows(self.df_csv)
        self.df_csv = self.append_column_y(self.df_csv)
        self.df_csv.set_index("y")
        self.num_classes = self.df_csv["y"].max() + 1
        _class_to_coord_list = self.get_class_to_coord_list()
        self.class_to_coord_map = torch.tensor(_class_to_coord_list)

        """ Build image cache """
        self.load_dataset_in_ram = load_dataset_in_ram
        self.image_cache = self._get_image_cache()

    def _get_image_cache(self):
        """Cache image paths or images itself"""
        image_cache = {}
        uuids = self.df_csv["uuid"].to_list()
        for uuid in uuids:
            image_dir = Path(self.path_images, uuid)
            image_filepaths = [Path(image_dir, "{}.jpg".format(degree)) for degree in self.degrees]
            cache_item = [Image.open(image_path) for image_path in image_filepaths] if self.load_dataset_in_ram else image_filepaths
            image_cache[uuid] = cache_item
        return image_cache

    def get_class_to_coord_list(self):
        _class_to_coord_map = []
        for row_idx in range(self.num_classes):
            row = self.df_csv.iloc[row_idx, :]
            true_lat, true_lng = row["latitude"], row["longitude"]
            point = [true_lat, true_lng]
            _class_to_coord_map.append(point)
        return _class_to_coord_map

    def name_without_extension(self, filename: Path | str):
        return Path(filename).stem

    def filter_df_rows(self, df: pd.DataFrame):
        """
        Args:
            df - dataframe
        Returns:
            Dataframe with rows for which the image exists
        """

        uuids_with_image = sorted(os.listdir(self.path_images))
        row_mask = df["uuid"].isin(uuids_with_image)
        return df.loc[row_mask, :]

    def append_column_y(self, df: pd.DataFrame):
        """
        y_map is temporary dataframe that hold non-duplicated values of polygon_index. y is then created by aranging polygon_index. The issue is that polygon_index might be interupted discrete series. The new column is uninterupted {0, ..., num_classes}

        Args:
            df - dataframe
        Returns:
            Dataframe with new column y
        """

        self.y_map = df.filter(["polygon_index"]).drop_duplicates().sort_values("polygon_index")
        self.y_map["y"] = np.arange(len(self.y_map))
        df = df.merge(self.y_map, on="polygon_index")
        return df

    def one_hot_encode_label(self, label: int):
        return one_hot_encode(label, self.num_classes)

    def get_row_attributes(self, row: pd.Series) -> Tuple[str, float, float, int, float, float, bool]:
        return str(row["uuid"]), row["latitude"], row["longitude"], int(row["y"]), row["centroid_lat"], row["centroid_lng"], bool(row["is_true_centroid"])

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, index: int):
        row = self.df_csv.iloc[index, :]
        uuid, image_latitude, image_longitude, label, centroid_lat, centroid_lng, is_true_centroid = self.get_row_attributes(row)

        images = self.image_cache[uuid]
        if not self.load_dataset_in_ram:
            images = [Image.open(image_path) for image_path in images]

        label = self.one_hot_encode_label(label)

        if self.image_transform is not None:
            transform = self.image_transform
            images = [transform(image) for image in images]
        if self.coordinate_transform is not None:
            transform = self.coordinate_transform
            image_latitude, image_longitude = self.coordinate_transform(image_latitude, image_longitude)

        image_coords = torch.tensor([image_latitude, image_longitude])
        return images, label, centroid_lat, centroid_lng, image_coords


if __name__ == "__main__":
    print("This file shouldn't be called as a script unless used for debugging.")
    dataset = GeoguesserDataset()
    print(dataset.__getitem__(2))
