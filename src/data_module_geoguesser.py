"""
GeoguesserDataModule preprocesses and hashes values that are shared acorss multiple Datasets (train/val/test). Any preprocessing that will be used across all datasets should be done here (scalings, caculating mean and std of images, hashing data values for faster item fetching...).

It handles the creation/loading of the main dataframe where images metadata is stored. The dataframe is passed to each Dataset. It also takes care of splitting the data (images) between different Datasets creating DataLoaders for each class.
"""
from __future__ import annotations, division

from itertools import combinations
from pathlib import Path
from typing import List, Optional, Union, Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import preprocess_csv_concat
import preprocess_csv_create_classes
from dataset_geoguesser import GeoguesserDataset
from defaults import DEAFULT_DROP_LAST, DEAFULT_NUM_WORKERS, DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING, \
    DEFAULT_LOAD_DATASET_IN_RAM, DEFAULT_SPACING, DEFAULT_TEST_FRAC, DEFAULT_TRAIN_FRAC, DEFAULT_VAL_FRAC
from utils_dataset import DatasetSplitType


class InvalidSizes(Exception):
    pass


class GeoguesserDataModule(pl.LightningDataModule):
    def __init__(
            self,
            cached_df: Path,
            dataset_dirs: List[Path],
            batch_size: int,
            train_frac=DEFAULT_TRAIN_FRAC,
            val_frac=DEFAULT_VAL_FRAC,
            test_frac=DEFAULT_TEST_FRAC,
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

        self.image_transform = image_transform
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.shuffle_before_splitting = shuffle_before_splitting

        """ Dataframe creation, numclasses handling and coord hashing"""
        self.df = self._handle_dataframe(cached_df) # data/complete/data__spacing_0.5__num_class_55.csv
        
        # TODO: extract this into a function
        lat_mean = self.df['latitude'].mean()
        lng_mean = self.df['longitude'].mean()
        lat_std = self.df['latitude'].std()
        lng_std = self.df['longitude'].std()
        
        # TODO: create a propert function instead of lambda function. It's too long to be in one line
        # TODO: saving the variable to self is not necesary as we don't use it anywhere else other than passing it to Datasets
        self.coords_transform = lambda lat, lng, lat_mean=lat_mean, lng_mean=lng_mean, lat_std=lat_std, lng_std=lng_std: torch.tensor([(lat - lat_mean) / lat_std, (lng - lng_mean) / lng_std]).float()

        self.num_classes = len(self.df["y"].drop_duplicates())
        assert self.num_classes == self.df["y"].max() + 1, "Number of classes should corespoing to the maximum y value of the csv dataframe"  # Sanity check
        
        self.class_to_centroid_map = torch.tensor(self._get_class_to_centroid_list(self.num_classes))

        self.train_dataset = GeoguesserDataset(
            df=self.df,
            num_classes=self.num_classes,
            dataset_dirs=self.dataset_dirs,
            image_transform=self.image_transform,
            load_dataset_in_ram=load_dataset_in_ram,
            dataset_type=DatasetSplitType.TRAIN,
            coordinate_transform=self.coords_transform
        )

        self.val_dataset = GeoguesserDataset(
            df=self.df,
            num_classes=self.num_classes,
            dataset_dirs=self.dataset_dirs,
            image_transform=self.image_transform,
            load_dataset_in_ram=load_dataset_in_ram,
            dataset_type=DatasetSplitType.VAL,
            coordinate_transform=self.coords_transform
        )

        self.test_dataset = GeoguesserDataset(
            df=self.df,
            num_classes=self.num_classes,
            dataset_dirs=self.dataset_dirs,
            image_transform=self.image_transform,
            load_dataset_in_ram=load_dataset_in_ram,
            dataset_type=DatasetSplitType.TEST,
            coordinate_transform=self.coords_transform
        )

    def _handle_dataframe(self, cached_df: Union[Path, None]):
        """
        Args:
            cached_df: path to the cached dataframe e.g. data/csv_decorated/data__spacing_0.2__num_class_231.csv

        - load the dataframe. If path is not provided the dataframe will be created in runtime (taking --dataset-dirs and --spacing into account)
        - remove rows with images that do not exist in the any of dataset directories
        - recount classes (y) because there might be polygons with no images assigned to them
            - note: the class count is same for all types (train/val/test) datasets
        """

        if cached_df:
            df = pd.read_csv(Path(cached_df))
        else:
            df_paths = [str(Path(dataset_dir, "data.csv")) for dataset_dir in self.dataset_dirs]
            df_merged = preprocess_csv_concat.main(["--csv", *df_paths, "--no-out"])
            df = preprocess_csv_create_classes.main(["--spacing", str(DEFAULT_SPACING), "--no-out"], df_merged)

        df = df[df["uuid"].isna() == False]  # remove rows for which the image doesn't exist
        map_poly_index_to_y = df.filter(["polygon_index"]).drop_duplicates().sort_values("polygon_index")
        map_poly_index_to_y["y"] = np.arange(len(map_poly_index_to_y))  # cols: polygon_index, y
        df = df.merge(map_poly_index_to_y, on="polygon_index")
        return df

    def _get_class_to_centroid_list(self, num_classes: int):

        """
        Args:
            num_classes: number of classes that were recounted ("y" column)
        Itterate over the information of each valid polygon/class and return it's centroids
        """

        df_class_info = self.df.loc[:,
                        ["polygon_index", "y", "centroid_lat", "centroid_lng", "is_true_centroid"]].drop_duplicates()
        _class_to_centroid_map = []
        for class_idx in range(num_classes):
            row = df_class_info.loc[df_class_info["y"] == class_idx].head(1)  # ensure that only one row is taken
            polygon_lat, polygon_lng = row["centroid_lat"].values[0], row["centroid_lng"].values[
                0]  # values -> ndarray with 1 dim
            point = [polygon_lat, polygon_lng]
            _class_to_centroid_map.append(point)
        return _class_to_centroid_map

    def _validate_sizes(self, train_frac, val_frac, test_frac):
        if sum([train_frac, val_frac, test_frac]) != 1:
            raise InvalidSizes("Sum of sizes has to be 1")

    def prepare_data(self) -> None:
        pass

    def _sanity_check_indices(self, dataset_train_indices: np.ndarray, dataset_val_indices: np.ndarray,
                              dataset_test_indices: np.ndarray):
        for ind_a, ind_b in combinations([dataset_train_indices, dataset_val_indices, dataset_test_indices], 2):
            assert len(np.intersect1d(ind_a, ind_b)) == 0, "Some indices share an index"
        set_ind = set(dataset_train_indices)
        set_ind.update(dataset_val_indices)
        set_ind.update(dataset_test_indices)
        assert len(set_ind) == (len(dataset_train_indices) + len(dataset_val_indices) + len(
            dataset_test_indices)), "Some indices might contain non-unqiue values"
        assert len(dataset_train_indices) > 0 and len(dataset_val_indices) > 0 and len(
            dataset_test_indices) > 0, "Some indices have no elements"

    def setup(self, stage: Optional[str] = None):

        dataset_train_indices = self.df.index[self.df["uuid"].isin(
            self.train_dataset.uuids)].to_list()  # type: ignore # [indices can be converted to list]
        dataset_val_indices = self.df.index[
            self.df["uuid"].isin(self.val_dataset.uuids)].to_list()  # type: ignore # [indices can be converted to list]
        dataset_test_indices = self.df.index[self.df["uuid"].isin(
            self.test_dataset.uuids)].to_list()  # type: ignore # [indices can be converted to list]
        self._sanity_check_indices(dataset_train_indices, dataset_val_indices, dataset_test_indices)

        if self.shuffle_before_splitting:
            np.random.shuffle(dataset_train_indices)

        self.train_size = len(dataset_train_indices)
        self.val_size = len(dataset_val_indices)
        self.test_size = len(dataset_test_indices)
        print("Train, val and test size", self.train_size, self.val_size, self.test_size)

        self.train_sampler = SubsetRandomSampler(dataset_train_indices)
        self.val_sampler = SubsetRandomSampler(dataset_val_indices)
        self.test_sampler = SubsetRandomSampler(dataset_test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=self.train_sampler, drop_last=self.drop_last, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=self.val_sampler, drop_last=self.drop_last, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=self.test_sampler, drop_last=self.drop_last, shuffle=False)


if __name__ == "__main__":
    pass
