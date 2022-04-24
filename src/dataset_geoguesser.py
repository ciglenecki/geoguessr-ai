from __future__ import annotations, division, print_function

from glob import glob
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from defaults import DEFAULT_LOAD_DATASET_IN_RAM
from utils_dataset import DatasetSplitType
from utils_functions import flatten, one_hot_encode


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
            dataset_dirs: List[Path],
            coordinate_transform: Callable = lambda lat, lng: torch.tensor([lat, lng]).float(),
            image_transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
            load_dataset_in_ram=DEFAULT_LOAD_DATASET_IN_RAM,
            dataset_type: DatasetSplitType = DatasetSplitType.TRAIN,
    ) -> None:
        print("GeoguesserDataset init")
        super().__init__()
        self.degrees = ["0", "90", "180", "270"]
        self.image_transform = image_transform
        self.coordinate_transform = coordinate_transform

        self.uuid_dir_paths = flatten(
            [glob(str(Path(dataset_dir, "images", dataset_type.value, "*"))) for dataset_dir in dataset_dirs])
        self.uuids = [Path(uuid_dir_path).stem for uuid_dir_path in self.uuid_dir_paths]
        self.df_csv = df
        self.num_classes = num_classes

        """ Build image cache """
        self.load_dataset_in_ram = load_dataset_in_ram
        self.image_cache = self._get_image_cache()

        self._sanity_check_images_dataframe()

    def _sanity_check_images_dataframe(self):
        size_rows_with_images = len(self.df_csv.loc[self.df_csv["uuid"].isin(self.uuids), :])
        assert size_rows_with_images == len(self.uuids), "Dataframe doesn't contain uuids for all images!"

    def _get_image_cache(self):
        """Cache image paths or images itself so that the __getitem__ function doesn't perform this job"""
        image_cache = {}
        for uuid, uuid_dir_path in zip(self.uuids, self.uuid_dir_paths):
            image_filepaths = [Path(uuid_dir_path, "{}.jpg".format(degree)) for degree in self.degrees]
            cache_item = [Image.open(image_path) for image_path in
                          image_filepaths] if self.load_dataset_in_ram else image_filepaths
            image_cache[uuid] = cache_item
        return image_cache

    def name_without_extension(self, filename: Path | str):
        return Path(filename).stem

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
        return str(row["uuid"]), float(row["latitude"]), float(row["longitude"]), int(row["y"])

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, index: int):
        row = self.df_csv.iloc[index, :]
        uuid, image_latitude, image_longitude, label = self._get_row_attributes(row)

        images = self.image_cache[uuid]
        if not self.load_dataset_in_ram:
            images = [Image.open(image_path) for image_path in images]

        label = self.one_hot_encode_label(label)

        images = [self.image_transform(image) for image in images]
        image_transformed_coords = self.coordinate_transform(image_latitude, image_longitude)
        return images, label, image_transformed_coords


if __name__ == "__main__":
    print("This file shouldn't be called as a script unless used for debugging.")
    dataset = GeoguesserDataset()
    print(dataset.__getitem__(2))
