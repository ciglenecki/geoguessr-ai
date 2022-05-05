from enum import Enum
from glob import glob
from itertools import product
from pathlib import Path
from typing import List, Union
from utils_functions import flatten, get_dirs_only

import pandas as pd

from utils_functions import flatten


class DatasetSplitType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PRED = "pred"


def filter_df_by_dataset_split(df: pd.DataFrame, dataset_dirs: List[Path], split_types: List[DatasetSplitType]):
    """Returns the dataframe with rows filtered by uuid. Only the uudis' from dataset_dirs with split_type=train/val/test are returned"""

    uuid_dir_paths = get_dataset_dirs_uuid_paths(dataset_dirs, split_types)
    uuids = [Path(uuid_dir_path).stem for uuid_dir_path in uuid_dir_paths]
    df_split_type = df.loc[df["uuid"].isin(uuids), :]
    return df_split_type


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
            get_dirs_only(Path(dataset_dir, "images", dataset_split_type.value))
            for dataset_dir, dataset_split_type in product(dataset_dirs, dataset_split_types)
        ]
    )
    return uuid_dir_paths
