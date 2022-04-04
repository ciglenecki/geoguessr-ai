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

# TODO: DOESNT MATTER FOR CLASSFICATION BECAUSE CLASSIFICATION CAN ONLY MAP TO EXISINTG CLASSES.if prediction is outside of croatia find the closes point on the croatia shape and map it to that point
# TODO: explicitly add classes which are defined as points on the border. distribution densitiy will be higher than uniform on the coastal part but thats okay
# TODO: intersect coastal squards with overlapping squares.
# TODO: every polygoin in dataframe can also have additional column that is called center. It doesnt have to be center of the polygon, it can be the edge of the country if the polygon goes outside of the boudns.


def test():
    return

    # pnt1 = Point(45.8150, 15.9819)
    # pnt2 = Point(42.6507, 18.0944)
    # a = [radians(_) for _ in [45.8150, 15.9819]]
    # b = [radians(_) for _ in [42.6507, 18.0944]]
    # print(haversine_distances([a], [b]) * 6371000)

    best_crs = "EPSG:4326"
    croatia_crs = "EPSG:3765"  # https://epsg.io/3765
    percentage_of_land_considered_a_block = 0.3

    country_shape = get_country_shape("HR")
    x_min, y_min, x_max, y_max = country_shape.total_bounds
    polygons = get_grid(x_min, y_min, x_max, y_max, spacing=0.1)
    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=best_crs)

    intersecting_polygons = []
    for polygon in grid.geometry:
        for country_polygon in country_shape.geometry:  # country_polygon could be an island, whole land...
            if polygon.intersects(country_polygon):
                if (polygon.intersection(country_polygon).area / polygon.area) >= percentage_of_land_considered_a_block:
                    intersecting_polygons.append(polygon)

    print(len(intersecting_polygons))
    intersecting_polygons_df = gpd.GeoDataFrame({"geometry": intersecting_polygons})
    base = country_shape.plot(color="green")
    intersecting_polygons_df.plot(ax=base, alpha=0.3, linewidth=0.2, edgecolor="black")
    plt.show()


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
            # TODO: somehow save polygons too

        print("NA poly", self.df_csv["label"].isnull().sum())
        # TODO: handle this better, more warnings etc.
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
        Gets the uuid
        Loads images via the uuid
        Loads latitude and longitude via the csv and uuid
        Applies transforms
        """
        row = self.df_csv.iloc[index, :]
        uuid, latitude, longitude, label = self.get_row_attributes(row)
        image_dir = Path(self.path_images, uuid)
        image_filepaths = list(map(lambda degree: Path(image_dir, "{}.jpg".format(degree)), self.degrees))
        images = list(map(lambda x: Image.open(x), image_filepaths))
        label = self.one_hot_encode_label(label)
        # polygon = row["polygon"]

        if self.image_transform is not None:
            transform = self.image_transform
            images = list(map(lambda i: transform(i), images))
        if self.coordinate_transform is not None:
            transform = self.coordinate_transform
            latitude, longitude = self.coordinate_transform(latitude, longitude)

        # TODO: implement multiimage support
        return images[0], label


if __name__ == "__main__":
    test()
    print("This file shouldn't be called as a script unless used for debugging.")
    dataset = GeoguesserDataset()
    dataset.prepare_lat_lng()
    # print(dataset.__getitem__(2))
