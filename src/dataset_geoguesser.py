from __future__ import annotations, division, print_function

import os
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import preprocess_csv_decorate as preprocess_csv_decorate
from defaults import DEFAULT_LOAD_DATASET_IN_RAM
from utils_dataset import DatasetSplitType
from utils_functions import one_hot_encode
from utils_paths import PATH_DATA_RAW


class GeoguesserDataset(Dataset):
    """
    Pytorch Dataset class which defines how elements are fetched from the soruce (directory)

    1. Holds pointers to the data (images and coordinates)
    2. Fetches them lazly when __getitem__ is called
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_classes,
        dataset_dir: Path = PATH_DATA_RAW,
        image_transform: None | transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        coordinate_transform: None | Callable = lambda x, y: np.array([x, y]).astype("float"),
        load_dataset_in_ram=DEFAULT_LOAD_DATASET_IN_RAM,
        dataset_type: DatasetSplitType = DatasetSplitType.TRAIN,
    ) -> None:
        print("GeoguesserDataset init")
        super().__init__()
        self.degrees = ["0", "90", "180", "270"]
        self.image_transform = image_transform
        self.coordinate_transform = coordinate_transform
        self.path_images = Path(dataset_dir, dataset_type.value)
        self.uuids = sorted(next(os.walk(self.path_images))[1])
        self.df_csv = df
        self.num_classes = num_classes

        """ Build image cache """
        self.load_dataset_in_ram = load_dataset_in_ram
        self.image_cache = self._get_image_cache()

    def _get_image_cache(self):
        """Cache image paths or images itself so that the __getitem__ function doesn't perform this job"""
        image_cache = {}
        for uuid in self.uuids:
            image_dir = Path(self.path_images, uuid)
            image_filepaths = [Path(image_dir, "{}.jpg".format(degree)) for degree in self.degrees]
            cache_item = [Image.open(image_path) for image_path in image_filepaths] if self.load_dataset_in_ram else image_filepaths
            image_cache[uuid] = cache_item
        return image_cache

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

    def _get_row_attributes(self, row: pd.Series) -> Tuple[str, float, float, int]:
        return str(row["uuid"]), row["latitude"], row["longitude"], int(row["y"])

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, index: int):
        row = self.df_csv.iloc[index, :]
        uuid, image_latitude, image_longitude, label = self._get_row_attributes(row)

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
        return images, label, image_coords


if __name__ == "__main__":
    print("This file shouldn't be called as a script unless used for debugging.")
    dataset = GeoguesserDataset()
    print(dataset.__getitem__(2))
