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

from utils_functions import one_hot_encode
from utils_paths import PATH_DATA_RAW
import coords_decorate_csv

# TODO: implement polygons on the border; these are addional classes which are expliclty defined. These classes might clash with already exising classes (polygons). How? There might be a polygon which is close to the border and overlaps the explicitly defined polygon. Solution is to remove the intersection so that polygons don't overlap. Polygon on the border (the one that is explicitly defined) should have prioirty over getting more surface area.

# TODO important: outside of Croatia bound classification; prediction gives softmax of values; weighted sum ends up in Bosna, what do we do? Solution: find the closest point on the border

# TODO: use haversine_distances in a loss function. haversine_distances acts just like residual. It might be useful to square the haversine_distances to get similar formula to MSE


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
    ) -> None:
        print("GeoguesserDataset init")
        super().__init__()
        self.degrees = ["0", "90", "180", "270"]
        self.image_transform = image_transform
        self.coordinate_transform = coordinate_transform
        self.path_images = Path(dataset_dir, "data")
        self.df_csv = pd.read_csv(Path(cached_df)) if cached_df else coords_decorate_csv.main(["--spacing", str(0.2), "--no-out"])

        """ Filter the dataframe, only include rows for images that exist and remove polygons with no data"""
        self.uuids_with_image = sorted(os.listdir(self.path_images))
        self.df_csv = self.df_csv.loc[self.df_csv["uuid"].isin(self.uuids_with_image), :]
        self.y_map = self.df_csv.filter(["true_label"]).drop_duplicates().sort_values("true_label")
        self.num_classes = len(self.y_map)
        self.y_map["y"] = np.arange(self.num_classes)
        self.df_csv = self.df_csv.merge(self.y_map, on="true_label")
        self.df_csv.set_index("y")

        unq_rows = self.df_csv.drop_duplicates(subset='y', keep=False, inplace=False)
        haver_list = []

        for row_idx in unq_rows['y']:
            row = self.df_csv.iloc[row_idx, :]
            true_lat, true_lng = row["latitude"], row["longitude"]
            haver_x = [true_lat, true_lng]
            haver_list.append(haver_x)
        self.haver_list = np.array(haver_list)

    def name_without_extension(self, filename: Path | str):
        return Path(filename).stem

    def one_hot_encode_label(self, label: int):
        return one_hot_encode(label, self.num_classes)

    def get_row_attributes(self, row: pd.Series) -> Tuple[str, float, float, int, float, float, bool]:
        return str(row["uuid"]), row["latitude"], row["longitude"], int(row["y"]), row["centroid_lat"], row["centroid_lng"], bool(row["is_true_centroid"])

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, index: int):
        row = self.df_csv.iloc[index, :]
        uuid, latitude, longitude, label, centroid_lat, centroid_lng, is_true_centroid = self.get_row_attributes(row)
        image_dir = Path(self.path_images, uuid)
        image_filepaths = [Path(image_dir, "{}.jpg".format(degree)) for degree in self.degrees]
        images = [Image.open(image_path) for image_path in image_filepaths]
        label = self.one_hot_encode_label(label)

        if self.image_transform is not None:
            transform = self.image_transform
            images = [transform(image) for image in images]
        if self.coordinate_transform is not None:
            transform = self.coordinate_transform
            latitude, longitude = self.coordinate_transform(latitude, longitude)

        return images, label, centroid_lat, centroid_lng


if __name__ == "__main__":
    print("This file shouldn't be called as a script unless used for debugging.")
    dataset = GeoguesserDataset()
    print(dataset.__getitem__(2))
