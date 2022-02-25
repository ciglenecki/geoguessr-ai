from __future__ import annotations, division
from torch.utils.data.sampler import SubsetRandomSampler

import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from utils_env import DEFAULT_TEST_SIZE
from utils_paths import *
from typing import Callable, Dict, List, Generic
from PIL import Image, ImageFile
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data.dataset import Dataset
import pytorch_lightning as pl


class GeoguesserDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path = PATH_DATA_RAW, batch_size: int = 32, test_size=DEFAULT_TEST_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # make assignments here (val/train/test split)
        # called on every process in DDP

    def train_dataloader(self):
        train_split = Dataset(...)
        return DataLoader(train_split)

    def val_dataloader(self):
        val_split = Dataset(...)
        return DataLoader(val_split)

    def test_dataloader(self):
        test_split = Dataset(...)
        return DataLoader(test_split)

    def teardown(self):
        pass
        # clean up after fit or test
        # called on every process in DDP


# class LoaderInitializer:
#     def __init__(self, dataset: Dataset, dataset_size=1, test_size=0.3):
#         self.dataset = dataset
#         self.dataset_size = dataset_size
#         self.dataset_length = int(len(dataset) * dataset_size)
#         self.dataset_indices = np.arange(self.dataset_length)
#         np.random.shuffle(self.dataset_indices)
#         test_split_index = int(np.floor(test_size * self.dataset_length))
#         train_indices, test_indices = self.dataset_indices[test_split_index:], self.dataset_indices[:test_split_index]

#         self.train_len = test_split_index
#         self.test_len = len(self.dataset_indices) - test_split_index

#         self.train_sampler = SubsetRandomSampler(train_indices)
#         self.test_sampler = SubsetRandomSampler(test_indices)

#     def get_loaders(self, batch_size=1, num_workers=1, drop_last=True):
#         train_dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, sampler=self.train_sampler, drop_last=drop_last)
#         test_dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, sampler=self.test_sampler, drop_last=drop_last)
#         return train_dataloader, test_dataloader, self.test_len, self.train_len
