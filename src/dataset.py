from __future__ import annotations, division, print_function

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
from utils_functions import one_hot_encode
from utils_geo import get_country_shape, get_grid, get_intersecting_polygons
from utils_paths import PATH_DATA_RAW, PATH_WORLD_BORDERS
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, box, MultiPoint, Point
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from utils_paths import PATH_DATA_RAW
from math import radians

# TODO: implement polygons on the border; these are addional classes which are expliclty defined. These classes might clash with already exising classes (polygons). How? There might be a polygon which is close to the border and overlaps the explicitly defined polygon. Solution is to remove the intersection so that polygons don't overlap. Polygon on the border (the one that is explicitly defined) should have prioirty over getting more surface area.

# TODO: outside of Croatia bound classification; prediction gives softmax of values; weighted sum ends up in Bosna, what do we do? Solution: find the closest point on the border

# TODO: every polygoin in dataframe can also have additional column that is called center. It doesnt have to be center of the polygon, it can be the edge of the country if the polygon's center goes outside of country's bounds

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
    ) -> None:
        super().__init__()
        self.image_transform = image_transform
        self.coordinate_transform = coordinate_transform
        self.path_images = Path(dataset_dir, "data")
        self.path_csv = Path(dataset_dir, "data.csv")
        self.df_csv = pd.read_csv(self.path_csv, index_col=False)
        self.df_csv["label"] = np.nan
        self.degrees = ["0", "90", "180", "270"]
        self.num_classes = 0
        """
        os.walk is a generator and calling next will get the first result in the form of a 3-tuple (dirpath, dirnames, filenames). The [1] index returns only the dirnames from that tuple.
        """
        # self.uuids = sorted(next(os.walk(self.path_images))[1])
        self.prepare_lat_lng()

    def prepare_data(self):
        pass

    def prepare_lat_lng(self):
        percentage_of_land_considered_a_block = 0

        country_shape = get_country_shape("HR")
        x_min, y_min, x_max, y_max = country_shape.total_bounds
        grid_polygons = get_grid(x_min, y_min, x_max, y_max, spacing=0.2)

        intersecting_polygons = get_intersecting_polygons(grid_polygons, country_shape.geometry, percentage_of_intersection_threshold=0)

        df_geo_csv = gpd.GeoDataFrame(
            self.df_csv,
            geometry=gpd.points_from_xy(
                self.df_csv.loc[:, "longitude"],
                self.df_csv.loc[:, "latitude"],
            ),
        )

        for i, polygon in enumerate(intersecting_polygons):
            self.df_csv.loc[df_geo_csv.within(polygon), "label"] = i
            # TODO: save polygons too; during the training process we will have to compare great-circle distance of the true polygon to the prediction; saving centroid of the polygon might be sufficient?

        # TODO: here df_csv is filtered. Rows of the dataset (images) whose location is not known are filtered out.
        # (label = 0 means that there we no polygons assigned to the picture). handle this better, more warnings etc.
        self.df_csv = self.df_csv.loc[~self.df_csv["label"].isnull(), :]
        self.num_classes = len(intersecting_polygons)
        print("num_classes", self.num_classes)

        intersecting_polygons_df = gpd.GeoDataFrame({"geometry": intersecting_polygons})
        base = country_shape.plot(color="green")
        intersecting_polygons_df.plot(ax=base, alpha=0.5, linewidth=0.2, edgecolor="red")
        plt.show()

    def name_without_extension(self, filename: Path | str):
        return Path(filename).stem

    def one_hot_encode_label(self, label: int):
        return one_hot_encode(label, self.num_classes)

    def get_row_attributes(self, row: pd.Series):
        return str(row["uuid"]), float(row["latitude"]), float(row["longitude"]), int(row["label"])

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, index: int):
        """
        Loads images via the index
        Loads latitude and longitude via the csv and uuid
        Applies transforms
        """
        row = self.df_csv.iloc[index, :]
        uuid, latitude, longitude, label = self.get_row_attributes(row)
        image_dir = Path(self.path_images, uuid)
        image_filepaths = list(map(lambda degree: Path(image_dir, "{}.jpg".format(degree)), self.degrees))
        images = list(map(lambda x: Image.open(x), image_filepaths))
        label = self.one_hot_encode_label(label)

        if self.image_transform is not None:
            transform = self.image_transform
            images = list(map(lambda i: transform(i), images))
        if self.coordinate_transform is not None:
            transform = self.coordinate_transform
            latitude, longitude = self.coordinate_transform(latitude, longitude)

        # TODO: implement multiimage support
        return images[0], label


if __name__ == "__main__":
    print("This file shouldn't be called as a script unless used for debugging.")
    dataset = GeoguesserDataset()
    dataset.prepare_lat_lng()
    # print(dataset.__getitem__(2))
