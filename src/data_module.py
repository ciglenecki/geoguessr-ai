from __future__ import annotations, division

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

from dataset import GeoguesserDatasetRaw
from utils_env import DEAFULT_DROP_LAST, DEAFULT_NUM_WORKERS, DEFAULT_BATCH_SIZE, DEFAULT_DATASET_FRAC, DEFAULT_TEST_SIZE, DEFAULT_TRAIN_SIZE, DEFAULT_VAL_SIZE
from utils_paths import PATH_DATA_RAW


class InvalidRatios(Exception):
    pass


"""
https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule

Datamodule's jobs
1. Download / tokenize / process.
2. Clean and (maybe) save to disk.
3. Load inside Dataset.
4. Apply transforms (rotate, tokenize, etc…).
5. Wrap inside a DataLoader.
"""


class GeoguesserDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path = PATH_DATA_RAW,
        dataset_frac=DEFAULT_DATASET_FRAC,
        batch_size: int = DEFAULT_BATCH_SIZE,
        train_size=DEFAULT_TRAIN_SIZE,
        val_size=DEFAULT_VAL_SIZE,
        test_size=DEFAULT_TEST_SIZE,
        image_transform: transforms.Compose | None = transforms.Compose([transforms.ToTensor()]),
        num_workers=DEAFULT_NUM_WORKERS,
        drop_last=DEAFULT_DROP_LAST,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_frac = dataset_frac
        self.batch_size = batch_size

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        self.image_transform = image_transform
        self.num_workers = num_workers
        self.drop_last = drop_last

    def prepare_data(self):
        pass

    def setup(self):
        self.dataset = GeoguesserDatasetRaw(root_dir=self.data_dir, image_transform=self.image_transform)
        self.dataset_length = int(len(self.dataset) * self.dataset_frac)
        self.dataset_indices = np.arange(self.dataset_length)
        np.random.shuffle(self.dataset_indices)

        train_indices, val_indicies, test_indices = self.split_by_ratio(self.dataset_indices, self.train_size, self.val_size, self.test_size)

        self.train_len = len(train_indices)
        self.val_len = len(val_indicies)
        self.test_len = len(test_indices)

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indicies)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def train_dataloader(self):
        sampler = self.train_sampler
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, drop_last=self.drop_last)

    def val_dataloader(self):
        sampler = self.val_sampler
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, drop_last=self.drop_last)

    def test_dataloader(self):
        sampler = self.test_sampler
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, drop_last=self.drop_last)

    def split_by_ratio(self, arr: np.ndarray, *ratios):
        if sum(ratios) != 1:
            raise InvalidRatios("Sum of ratios must be equal to 1", ratios)
        ind = np.add.accumulate(np.array(ratios) * len(arr)).astype(int)
        return [x.tolist() for x in np.split(arr, ind)][: len(ratios)]


if __name__ == "__main__":
    print("This file shouldn't be called as a script unless used for debugging.")
    module = GeoguesserDataModule()
    module.setup()
    train_loader = module.train_dataloader()
    for i in train_loader:
        print(i)
        break
