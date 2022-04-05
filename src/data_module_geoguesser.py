from __future__ import annotations, division

from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torchvision import transforms

from dataset import GeoguesserDataset
from utils_env import DEAFULT_DROP_LAST, DEAFULT_NUM_WORKERS, DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING, \
    DEFAULT_BATCH_SIZE, DEFAULT_TEST_FRAC, DEFAULT_TRAIN_FRAC, DEFAULT_VAL_FRAC
from utils_functions import split_by_ratio
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
    ) -> None:
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
        print(dataset_dir)

        self.dataset = GeoguesserDataset(dataset_dir=self.dataset_dir, image_transform=self.image_transform)

    def prepare_data(self) -> None:
        pass

    def _validate_sizes(
            self,
            train_frac,
            val_frac,
            test_frac,
    ):
        if sum([train_frac, val_frac, test_frac]) != 1:
            raise InvalidSizes("Sum of sizes has to be 1")

    def setup(self, stage: Optional[str] = None):

        """
        dataset_indices = [0,1,2,3,4,5,6,7,8,9]
        TODO: should dataset_indices be shuffled?
        """

        dataset_indices = np.arange(len(self.dataset))

        if self.shuffle_before_splitting:
            np.random.shuffle(dataset_indices)

        dataset_train_indices, dataset_val_indices, dataset_test_indices = split_by_ratio(dataset_indices,
                                                                                          self.train_frac,
                                                                                          self.val_frac,
                                                                                          self.train_frac,
                                                                                          use_whole_array=True)

        self.train_size = len(dataset_train_indices)
        self.val_size = len(dataset_val_indices)
        self.test_size = len(dataset_test_indices)

        self.train_sampler = SubsetRandomSampler(dataset_train_indices)
        self.val_sampler = SubsetRandomSampler(dataset_val_indices)
        self.test_sampler = SubsetRandomSampler(dataset_test_indices)

    def train_dataloader(self):
        return self._get_common_dataloader(self.train_sampler, shuffle=True)

    def val_dataloader(self):
        return self._get_common_dataloader(self.val_sampler, shuffle=False)

    def test_dataloader(self):
        return self._get_common_dataloader(self.test_sampler, shuffle=False)

    def _get_common_dataloader(self, sampler: Sampler[int], shuffle: bool):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler,
                          drop_last=self.drop_last, shuffle=False)


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
