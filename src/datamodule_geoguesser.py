"""
GeoguesserDataModule preprocesses and hashes values that are shared acorss multiple Datasets (train/val/test). Any preprocessing that will be used across all datasets should be done here (scalings, caculating mean and std of images, hashing data values for faster item fetching...).

It handles the creation/loading of the main dataframe where images metadata is stored. The dataframe is passed to each Dataset. It also takes care of splitting the data (images) between different Datasets creating DataLoaders for each class.
"""
from __future__ import annotations, division

import os
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import preprocess_csv_concat
import preprocess_csv_create_polygons
from dataset_geoguesser import GeoguesserDataset, GeoguesserDatasetPredict
from defaults import (
    DEAFULT_DROP_LAST,
    DEAFULT_NUM_WORKERS,
    DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_FRAC,
    DEFAULT_LOAD_DATASET_IN_RAM,
    DEFAULT_SPACING,
    DEFAULT_TEST_FRAC,
    DEFAULT_TRAIN_FRAC,
    DEFAULT_VAL_FRAC,
)
from utils_dataset import DatasetSplitType, filter_df_by_dataset_split
from utils_functions import print_df_sample
from utils_paths import PATH_DATA_COMPLETE, PATH_DATA_EXTERNAL, PATH_DATA_RAW


class InvalidSizes(Exception):
    pass


class GeoguesserDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cached_df: Path,
        dataset_dirs: List[Path],
        batch_size: int = DEFAULT_BATCH_SIZE,
        train_frac=DEFAULT_TRAIN_FRAC,
        val_frac=DEFAULT_VAL_FRAC,
        test_frac=DEFAULT_TEST_FRAC,
        dataset_frac=DEFAULT_DATASET_FRAC,
        image_transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        num_workers=DEAFULT_NUM_WORKERS,
        drop_last=DEAFULT_DROP_LAST,
        shuffle_before_splitting=DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING,
        load_dataset_in_ram=DEFAULT_LOAD_DATASET_IN_RAM,
    ) -> None:
        super().__init__()
        print("GeoguesserDataModule init")

        self._validate_sizes(train_frac, val_frac, test_frac)

        self.dataset_dirs = dataset_dirs
        self.batch_size = batch_size

        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.dataset_frac = dataset_frac

        self.image_transform = image_transform
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.shuffle_before_splitting = shuffle_before_splitting

        """ Dataframe loading, numclasses handling and min max scaling"""
        df = self._load_dataframe(cached_df)
        df = self._dataframe_create_classes(df)
        df = self._adding_centroids_weighted(df)
        self.crs_scaler = self._get_and_fit_min_max_scaler_for_train_data(df)
        df = self._scale_min_max_crs_columns(df, self.crs_scaler)
        self.df = df
        print_df_sample(self.df)

        """ Creating CRS hash map for all classes"""
        self.num_classes = len(self.df["y"].drop_duplicates())
        assert (
            self.num_classes == self.df["y"].max() + 1
        ), "Number of classes should corespoing to the maximum y value of the csv dataframe"  # Sanity check
        (
            self.class_to_crs_centroid_map,
            self.class_to_latlng_centroid_map,
            self.class_to_crs_weighted_map,
        ) = self._get_class_to_coords_maps(self.num_classes)

        self.train_dataset = GeoguesserDataset(
            df=self.df,
            num_classes=self.num_classes,
            dataset_dirs=self.dataset_dirs,
            image_transform=self.image_transform,
            load_dataset_in_ram=load_dataset_in_ram,
            dataset_type=DatasetSplitType.TRAIN,
        )

        self.val_dataset = GeoguesserDataset(
            df=self.df,
            num_classes=self.num_classes,
            dataset_dirs=self.dataset_dirs,
            image_transform=self.image_transform,
            load_dataset_in_ram=load_dataset_in_ram,
            dataset_type=DatasetSplitType.VAL,
        )

        self.test_dataset = GeoguesserDataset(
            df=self.df,
            num_classes=self.num_classes,
            dataset_dirs=self.dataset_dirs,
            image_transform=self.image_transform,
            load_dataset_in_ram=load_dataset_in_ram,
            dataset_type=DatasetSplitType.TEST,
        )

    def _load_dataframe(self, cached_df: Union[Path, None]) -> pd.DataFrame:
        """
        Returns the cached dataframe if the path file is given. If not, dataframe is created in runtime (taking --dataset-dirs and --spacing into account) and returned either way.

        Args:
            cached_df: e.g. data/csv_decorated/data__spacing_0.2__num_class_231.csv
        """
        if cached_df:
            df = pd.read_csv(Path(cached_df))
        else:
            df_paths = [str(Path(dataset_dir, "data.csv")) for dataset_dir in self.dataset_dirs]
            df_merged = preprocess_csv_concat.main(["--csv", *df_paths, "--no-out"])
            df = preprocess_csv_create_polygons.main(["--spacing", str(DEFAULT_SPACING), "--no-out"], df_merged)

        return df

    def _dataframe_create_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove Dataframe rows for images images that do not exist in the any of dataset directories.
        Then,  recount classes (y) because there might be polygons with no images assigned to them
        note: the class count is same for all types (train/val/test) datasets

        Args:
            df: dataframe
        """

        df = df[df["uuid"].isna() == False]  # remove rows for which the image doesn't exist
        map_poly_index_to_y = df.filter(["polygon_index"]).drop_duplicates().sort_values("polygon_index")
        map_poly_index_to_y["y"] = np.arange(len(map_poly_index_to_y))  # cols: polygon_index, y
        df = df.merge(map_poly_index_to_y, on="polygon_index")
        return df

    def _get_and_fit_min_max_scaler_for_train_data(self, df: pd.DataFrame) -> MinMaxScaler:
        """
        Scales all crs columns to [0, 1] using only the training data

        Args:
            df: dataframe
        """
        df_train = filter_df_by_dataset_split(df, self.dataset_dirs, DatasetSplitType.TRAIN)
        crs_scaler = MinMaxScaler()
        crs_scaler.fit(df_train.loc[:, ["crs_x", "crs_y"]])
        return crs_scaler

    def _adding_centroids_weighted(self, df: pd.DataFrame) -> pd.DataFrame:
        df_train = filter_df_by_dataset_split(df, self.dataset_dirs, DatasetSplitType.TRAIN)
        df_train["lat_weighted"] = df_train["latitude"].groupby(df_train["y"]).transform("mean")
        df_train["lng_weighted"] = df_train["longitude"].groupby(df_train["y"]).transform("mean")
        df_train["crs_y_weighted"] = df_train["crs_y"].groupby(df_train["y"]).transform("mean")
        df_train["crs_x_weighted"] = df_train["crs_x"].groupby(df_train["y"]).transform("mean")
        df_filter = df_train.filter(
            items=[
                "y",
                "lat_weighted",
                "lng_weighted",
                "crs_y_weighted",
                "crs_x_weighted",
            ]
        ).drop_duplicates()
        df = df.merge(df_filter, on="y")
        return df

    def _scale_min_max_crs_columns(self, df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
        """
        Scales all `crs` columns to [0, 1] using the scaler that was fit on training data

        Args:
            df: dataframe
        """
        df.loc[:, ["crs_x_minmax", "crs_y_minmax"]] = scaler.transform(df.loc[:, ["crs_x", "crs_y"]])

        df.loc[:, ["crs_centroid_x_minmax", "crs_centroid_y_minmax"]] = scaler.transform(
            df.loc[:, ["crs_centroid_x", "crs_centroid_y"]]
        )
        df.loc[:, ["crs_x_weighted_minmax", "crs_y_weighted_minmax"]] = scaler.transform(
            df.loc[:, ["crs_x_weighted", "crs_y_weighted"]]
        )

        return df

    def _get_class_to_coords_maps(self, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns list of tuples (lat,lng). Index of the element in the list (class_idx) defines the class and the element (tuple) defines the CRS minmax centroid of the class. note: in the for loop, we take only 1 row with concrete class class. Then we extract the coords from that row.

        Args:
            num_classes: number of classes that were recounted ("y" column)
        Itterate over the information of each valid polygon/class and return it's centroids.

        """

        df_class_info = self.df.loc[
            :,
            [
                "polygon_index",
                "y",
                "crs_centroid_x_minmax",
                "crs_centroid_y_minmax",
                "centroid_lat",
                "centroid_lng",
                "crs_x_weighted_minmax",
                "crs_y_weighted_minmax",
            ],
        ].drop_duplicates()

        class_to_crs_centroid_map = []
        class_to_latlng_centroid_map = []
        class_to_crs_weighted_map = []

        for class_idx in range(num_classes):
            row = df_class_info.loc[df_class_info["y"] == class_idx].head(1)  # ensure that only one row is taken
            polygon_crs_x, polygon_crs_y = (
                row["crs_centroid_x_minmax"].values[0],
                row["crs_centroid_y_minmax"].values[0],
            )  # values extracts values as numpy array

            polygon_crs_x_weighted, polygon_crs_y_weighted = (
                row["crs_x_weighted_minmax"].values[0],
                row["crs_y_weighted_minmax"].values[0],
            )  # values extracts values as numpy array

            polygon_lat, polygon_lng = (
                row["centroid_lat"].values[0],
                row["centroid_lng"].values[0],
            )  # values extracts values as numpy array
            class_to_crs_centroid_map.append([polygon_crs_x, polygon_crs_y])
            class_to_latlng_centroid_map.append([polygon_lat, polygon_lng])
            class_to_crs_weighted_map.append([polygon_crs_x_weighted, polygon_crs_y_weighted])

        return (
            torch.tensor(class_to_crs_centroid_map),
            torch.tensor(class_to_latlng_centroid_map),
            torch.tensor(class_to_crs_weighted_map),
        )

    def store_df_to_report(self, path: Path):
        os.makedirs(path.parents[0])
        self.df.to_csv(path, mode="w+", index=True, header=True)

    def _validate_sizes(self, train_frac, val_frac, test_frac):
        if sum([train_frac, val_frac, test_frac]) != 1:
            raise InvalidSizes("Sum of sizes has to be 1")

    def prepare_data(self) -> None:
        pass

    def _sanity_check_indices(
        self,
        dataset_train_indices: np.ndarray,
        dataset_val_indices: np.ndarray,
        dataset_test_indices: np.ndarray,
    ):
        for ind_a, ind_b in combinations([dataset_train_indices, dataset_val_indices, dataset_test_indices], 2):
            assert len(np.intersect1d(ind_a, ind_b)) == 0, "Some indices share an index {}".format(
                np.intersect1d(ind_a, ind_b)
            )
        set_ind = set(dataset_train_indices)
        set_ind.update(dataset_val_indices)
        set_ind.update(dataset_test_indices)
        assert len(set_ind) == (
            len(dataset_train_indices) + len(dataset_val_indices) + len(dataset_test_indices)
        ), "Some indices might contain non-unqiue values"
        assert (
            len(dataset_train_indices) > 0 and len(dataset_val_indices) > 0 and len(dataset_test_indices) > 0
        ), "Some indices have no elements"

    def setup(self, stage: Optional[str] = None):

        dataset_train_indices = self.df.index[
            self.df["uuid"].isin(self.train_dataset.uuids)
        ].to_numpy()  # type: ignore # [indices can be converted to list]
        dataset_val_indices = self.df.index[
            self.df["uuid"].isin(self.val_dataset.uuids)
        ].to_numpy()  # type: ignore # [indices can be converted to list]
        dataset_test_indices = self.df.index[
            self.df["uuid"].isin(self.test_dataset.uuids)
        ].to_numpy()  # type: ignore # [indices can be converted to list]

        if self.dataset_frac != 1:
            dataset_train_indices = np.random.choice(
                dataset_train_indices, int(self.dataset_frac * len(dataset_train_indices)), replace=False
            )
            dataset_val_indices = np.random.choice(
                dataset_val_indices, int(self.dataset_frac * len(dataset_val_indices)), replace=False
            )
            dataset_test_indices = np.random.choice(
                dataset_test_indices, int(self.dataset_frac * len(dataset_test_indices)), replace=False
            )

        self._sanity_check_indices(dataset_train_indices, dataset_val_indices, dataset_test_indices)

        if self.shuffle_before_splitting:
            np.random.shuffle(dataset_train_indices)

        self.train_size = len(dataset_train_indices)
        self.val_size = len(dataset_val_indices)
        self.test_size = len(dataset_test_indices)

        self.train_sampler = SubsetRandomSampler(dataset_train_indices)
        self.val_sampler = SubsetRandomSampler(dataset_val_indices)
        self.test_sampler = SubsetRandomSampler(dataset_test_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            drop_last=self.drop_last,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            drop_last=self.drop_last,
        )


class GeoguesserDataModulePredict(pl.LightningDataModule):
    def __init__(
        self,
        images_dirs: List[Path],
        num_classes: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        image_transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        num_workers=DEAFULT_NUM_WORKERS,
    ) -> None:
        super().__init__()
        print("GeoguesserDataModule init")

        self.image_transform = image_transform
        self.num_workers = num_workers
        self.drop_last = False
        self.batch_size = batch_size
        self.num_classes = num_classes  # TODO
        self.predict_dataset = GeoguesserDatasetPredict(
            images_dirs=images_dirs,
            num_classes=self.num_classes,
        )

    def _validate_sizes(self, train_frac, val_frac, test_frac):
        if sum([train_frac, val_frac, test_frac]) != 1:
            raise InvalidSizes("Sum of sizes has to be 1")

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            shuffle=False,
        )


if __name__ == "__main__":
    dm = GeoguesserDataModule(
        cached_df=Path(PATH_DATA_COMPLETE, "data__spacing_0.5__num_class_55.csv"),
        dataset_dirs=[PATH_DATA_RAW],
    )
    dm.setup()
    pass
