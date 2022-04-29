from enum import Enum
from glob import glob
from pathlib import Path
from typing import List, Union
from itertools import product
from utils_functions import flatten
import pandas as pd


class DatasetSplitType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def filter_df_by_dataset_split(df: pd.DataFrame, dataset_dirs: List[Path], split_type: DatasetSplitType):
    """Returns the dataframe with rows filtered by uuid. Only the uudis' from dataset_dirs with split_type=train/val/test are returned"""
    uuid_dir_paths = flatten(
        [glob(str(Path(dataset_dir, "images", split_type.value, "*"))) for dataset_dir in dataset_dirs]
    )
    uuids = [Path(uuid_dir_path).stem for uuid_dir_path in uuid_dir_paths]
    df_train = df.loc[df["uuid"].isin(uuids)]
    return df_train


def get_dataset_dirs_uuid_paths(
    dataset_dirs: Union[List[Path], Path], dataset_split_types: Union[List[DatasetSplitType], DatasetSplitType]
):
    """Returns all uuids from dataset directories for a given dataset split type
    e.g.
        dataset_dirs = ['data/raw', 'data/external']
        dataset_split_type = DatasetSplitType.TRAIN
    returns all uuids for train images in those directories
    """

    if not isinstance(dataset_dirs, list):
        dataset_dirs = [dataset_dirs]
    if not isinstance(dataset_split_types, list):
        dataset_split_types = [dataset_split_types]

    uuid_dir_paths = flatten(
        [
            glob(str(Path(dataset_dir, "images", dataset_split_type.value, "*")))
            for dataset_dir, dataset_split_type in product(dataset_dirs, dataset_split_types)
        ]
    )
    return uuid_dir_paths
