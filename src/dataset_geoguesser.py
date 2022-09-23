from __future__ import annotations, division, print_function

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils_dataset import DatasetSplitType, get_dataset_dirs_uuid_paths
from utils_functions import flatten, get_dirs_only, one_hot_encode


class GeoguesserDataset(Dataset):
    """
    Pytorch Dataset class which defines how elements are fetched from the source (directory)

    1. Holds pointers to the data (images and coordinates)
    2. Fetches them lazily when __getitem__ is called
    3. Image and coord transformations are applied before __getitem__ returns image and coordinates
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_classes,
        dataset_dirs: list[Path],
        crs_coords_transform: Callable = lambda crs_x, crs_y: torch.tensor(
            [crs_x, crs_y]
        ).float(),
        image_transform: transforms.Compose = transforms.Compose(
            [transforms.ToTensor()]
        ),
        dataset_type: DatasetSplitType = DatasetSplitType.TRAIN,
    ) -> None:
        print("GeoguesserDataset {} init".format(dataset_type.value))
        super().__init__()

        self.degrees = ["0", "90", "180", "270"]
        self.image_transform = image_transform
        self.crs_coords_transform = crs_coords_transform
        self.num_classes = num_classes

        """Any rows that contain NaN and have uuid will be used to remove the uuid and images from the dataset. These are invalid locations for which the projection couldn't be caculated."""

        """ Populate uuid, uuid path and image path variables """
        uuid_dir_paths = get_dataset_dirs_uuid_paths(
            dataset_dirs=dataset_dirs, dataset_split_types=dataset_type
        )

        self.uuids = []
        self.image_filepaths = []
        self.image_store = {}

        for uuid_dir_path in uuid_dir_paths:
            uuid = Path(uuid_dir_path).stem
            image_filepaths = [
                Path(uuid_dir_path, "{}.jpg".format(degree)) for degree in self.degrees
            ]
            self.image_store[uuid] = image_filepaths
            self.uuids.append(uuid)
            self.image_filepaths.append(image_filepaths)
        self.df_csv = df.loc[df["uuid"].isin(self.uuids), :]
        self._sanity_check_images_dataframe()

    def _sanity_check_images_dataframe(self):
        assert not self.df_csv.isnull().any().any(), "Dataframe contains NaN values"
        set_inner = set(self.df_csv["uuid"].to_list())
        set_outer = set(self.image_store.keys())
        assert set_inner.issubset(
            set_outer
        ), "Dataframe shoudln't contain uuids which are not in image_store (image directory)"

    def append_column_y(self, df: pd.DataFrame):
        """
        y_map is temporary dataframe that hold non-duplicated values of polygon_index. y is then created by aranging polygon_index. The issue is that polygon_index might be interupted discrete series. The new column is uninterupted {0, ..., num_classes}

        Args:
            df - dataframe
        Returns:
            Dataframe with new column y
        """

        self.y_map = (
            df.filter(["polygon_index"]).drop_duplicates().sort_values("polygon_index")
        )
        self.y_map["y"] = np.arange(len(self.y_map))
        df = df.merge(self.y_map, on="polygon_index")
        return df

    def one_hot_encode_label(self, label: int):
        return one_hot_encode(label, self.num_classes)

    def _get_row_attributes(self, row: pd.Series) -> Tuple[str, float, float, int]:
        return (
            str(row["uuid"]),
            float(row["crs_x_minmax"]),
            float(row["crs_y_minmax"]),
            int(row["y"]),
        )

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, index: int):
        row = self.df_csv.loc[index, :]
        uuid, crs_x, crs_y, label = self._get_row_attributes(row)
        image_paths = self.image_store[uuid]
        images = [Image.open(image_path) for image_path in image_paths]

        label = self.one_hot_encode_label(label)

        images = [self.image_transform(image) for image in images]
        crs_coords = self.crs_coords_transform(crs_x, crs_y)
        return images, label, crs_coords


class GeoguesserDatasetPredict(Dataset):
    def __init__(
        self,
        images_dirs: list[Path],
        num_classes: int,
        image_transform: transforms.Compose = transforms.Compose(
            [transforms.ToTensor()]
        ),
    ) -> None:
        print("GeoguesserDatasetPredict init")
        super().__init__()
        self.num_classes = num_classes
        self.degrees = ["0", "90", "180", "270"]
        self.image_transform = image_transform
        self.uuid_dir_paths = flatten(
            [get_dirs_only(images_dir) for images_dir in images_dirs]
        )
        self.uuids = [Path(uuid_dir_path).stem for uuid_dir_path in self.uuid_dir_paths]

        """ Build image cache """
        self.image_store = self._get_image_store()

    def _get_image_store(self):
        image_store = {}
        for uuid, uuid_dir_path in zip(self.uuids, self.uuid_dir_paths):
            image_filepaths = [
                Path(uuid_dir_path, "{}.jpg".format(degree)) for degree in self.degrees
            ]
            cache_item = image_filepaths
            image_store[uuid] = cache_item
        return image_store

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, index: int):

        uuid = self.uuids[index]
        images = [Image.open(image_path) for image_path in self.image_store[uuid]]
        images = [self.image_transform(image) for image in images]
        return images, uuid


if __name__ == "__main__":
    print("This file shouldn't be called as a script unless used for debugging.")
    dataset = GeoguesserDataset()
    print(dataset.__getitem__(2))
