"""
File that filters the rows of the csv dataframe. Rows are filtered to only include uuids from the dataset directory (--data-dir). 

E.g.
    Base CSV:
        uuid,data
        -----------
        ab,"test"
        ba2,"tesa"
        dxa,"teab1"

    Dataset directory:
    ├──ab/
    └──dxa

Saves the csv:
    uuid,data
    -----------
    ab,"test"
    dxa,"teab1"
"""

from pathlib import Path
import pandas as pd
from utils_paths import PATH_DATA_EXTERNAL, PATH_DATA_SAMPLER
import os
import argparse
from utils_functions import is_valid_dir
import sys


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-csv",
        metavar="file.csv",
        help="Base-line csv that contains uuid, lat and lng information for images of the external source. E.g. data/coord_sampler/coords_sample__n_1000000_modified_22-04-17-17-28-25.csv",
        default=Path(PATH_DATA_SAMPLER, "coords_sample__n_1000000_modified_22-04-17-17-28-25.csv"),
    )

    parser.add_argument(
        "--data-dir",
        metavar="dir",
        type=is_valid_dir,
        help="Directory with images from the external source",
        default=Path(PATH_DATA_EXTERNAL, "data"),
    )
    parser.add_argument(
        "--out",
        metavar="csv",
        help="Path of the csv output",
        default=Path(PATH_DATA_EXTERNAL, "data_external.csv"),
    )
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    uuids = os.listdir(args.data_dir)
    df = pd.read_csv(args.base_csv)
    df = df.loc[df["uuid"].isin(uuids), :]
    print("Saving dataframe with {} rows to {}".format(len(df), args.out))
    df.to_csv(args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
