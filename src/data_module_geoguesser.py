from __future__ import annotations, division

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from dataset import GeoguesserDataset
from utils_env import DEAFULT_DROP_LAST, DEAFULT_NUM_WORKERS, DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING, DEFAULT_BATCH_SIZE, DEFAULT_TEST_FRAC, DEFAULT_TRAIN_FRAC, DEFAULT_VAL_FRAC
from utils_functions import split_by_ratio
from utils_env import DEAFULT_DROP_LAST, DEAFULT_NUM_WORKERS, DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING, \
    DEFAULT_BATCH_SIZE, DEFAULT_LOAD_DATASET_IN_RAM, DEFAULT_TEST_FRAC, DEFAULT_TRAIN_FRAC, DEFAULT_VAL_FRAC
from utils_paths import PATH_DATA_RAW


class InvalidSizes(Exception):
    pass


class GeoguesserDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: Path = PATH_DATA_RAW,
        batch_size: int = DEFAULT_BATCH_SIZE,
        train_frac=DEFAULT_TRAIN_FRAC,
        val_frac=DEFAULT_VAL_FRAC,
        test_frac=DEFAULT_TEST_FRAC,
        image_transform: None | transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        num_workers=DEAFULT_NUM_WORKERS,
        drop_last=DEAFULT_DROP_LAST,
        shuffle_before_splitting=DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING,
        cached_df=None,
    ) -> None:
        self.test_sampler = None
        self.val_sampler = None
        self.train_sampler = None
        self.test_size = None
        self.val_size = None
        self.train_size = None
        print("GeoguesserDataModule init")
        super().__init__()

        self._validate_sizes(train_frac, val_frac, test_frac)

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

        self.image_transform = image_transform
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.shuffle_before_splitting = shuffle_before_splitting

        self.train_dataset = GeoguesserDataset(
            dataset_dir=self.dataset_dir,
            image_transform=self.image_transform,
            cached_df=cached_df,
            load_dataset_in_ram=load_dataset_in_ram,
            dataset_type="train"
        )

        self.val_dataset = GeoguesserDataset(
            dataset_dir=self.dataset_dir,
            image_transform=self.image_transform,
            cached_df=cached_df,
            load_dataset_in_ram=load_dataset_in_ram,
            dataset_type="val"
        )

        self.test_dataset = GeoguesserDataset(
            dataset_dir=self.dataset_dir,
            image_transform=self.image_transform,
            cached_df=cached_df,
            load_dataset_in_ram=load_dataset_in_ram,
            dataset_type="test"
        )

        num_classes = len(pd.unique(np.concatenate((self.train_dataset.df_csv['y'], self.val_dataset.df_csv['y']), axis=0)))
        self.train_dataset.num_classes = num_classes
        self.val_dataset.num_classes = num_classes

        # _class_to_coord_list = self.train_dataset.get_class_to_coord_list()
        # self.train_dataset.class_to_coord_map = torch.tensor(_class_to_coord_list)
        #
        # _class_to_coord_list = self.val_dataset.get_class_to_coord_list()
        # self.val_dataset.class_to_coord_map = torch.tensor(_class_to_coord_list)

    def prepare_data(self) -> None:
        pass

    def _validate_sizes(self, train_frac, val_frac, test_frac):
        if sum([train_frac, val_frac, test_frac]) != 1:
            raise InvalidSizes("Sum of sizes has to be 1")

    def setup(self, stage: Optional[str] = None):

        dataset_train_indices = np.arange(len(self.train_dataset))
        dataset_val_indices = np.arange(len(self.val_dataset))
        dataset_test_indices = np.arange(len(self.test_dataset))

        if self.shuffle_before_splitting:
            np.random.shuffle(dataset_train_indices)
            np.random.shuffle(dataset_val_indices)
            np.random.shuffle(dataset_test_indices)

        # dataset_train_indices, dataset_val_indices, dataset_test_indices = split_by_ratio(dataset_indices, self.train_frac, self.val_frac, self.test_frac, use_whole_array=True)

        self.train_size = len(dataset_train_indices)
        self.val_size = len(dataset_val_indices)
        self.test_size = len(dataset_test_indices)

        self.train_sampler = SubsetRandomSampler(dataset_train_indices)
        self.val_sampler = SubsetRandomSampler(dataset_val_indices)
        self.test_sampler = SubsetRandomSampler(dataset_test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=self.train_sampler,
                          drop_last=self.drop_last,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=self.val_sampler,
                          drop_last=self.drop_last,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=self.test_sampler,
                          drop_last=self.drop_last,
                          shuffle=False)

    # def _get_common_dataloader(self, sampler: Sampler[int], shuffle: bool):
    #     return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, drop_last=self.drop_last, shuffle=False)


if __name__ == "__main__":
    pass

# if __name__ == "__main__":
#     print("This file shouldn't be called as a script unless used for debugging.")
#     module = GeoguesserDataModule()
#     module.setup()
#     train_loader = module.train_dataloader()
#     for i in train_loader:
#         print(i)
#         break
