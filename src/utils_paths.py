"""
Specified paths -- directory structure

WORK_DIR is defined as one level below this file
"""

import os
from pathlib import Path

"""
parents attribute contains all the parent directories of a given path
- p.parents[0] is the directory containing file
- p.parents[1] will be the next directory up
"""

WORK_DIR = Path(os.path.realpath(__file__)).parents[1]
PATH_DATA = Path(WORK_DIR, "data")

PATH_DATA_COMPLETE = Path(PATH_DATA, "dataset_complete")

"""DATA/ORIGINAL"""
PATH_DATA_ORIGINAL = Path(PATH_DATA, "original")

"""DATA/EXTERNAL"""
PATH_DATA_EXTERNAL = Path(PATH_DATA, "external")

"""DATA/EXAMPLE"""
PATH_DATA_EXAMPLE = Path(PATH_DATA, "example")
PATH_DATA_EXAMPLE_IMAGES = Path(PATH_DATA_EXTERNAL, "images")
PATH_DATA_EXAMPLE = Path(PATH_DATA, "original")


"""SUBSETS"""
PATH_DATA_SUBSET_EXTERNAL = Path(PATH_DATA, "dataset_external_subset")
PATH_DATA_SUBSET_ORIGINAL = Path(PATH_DATA, "dataset_original_subset")

PATH_DATA_SAMPLER = Path(PATH_DATA, "coord_sampler")
PATH_WORLD_BORDERS = Path(PATH_DATA, "world-borders", "world_no_croatia_islands.shp")

PATH_REPORT = Path(WORK_DIR, "reports")
PATH_REPORT_QUICK = Path(WORK_DIR, "reports-quick")
PATH_FIGURE = Path(WORK_DIR, "figures")
PATH_MODEL = Path(WORK_DIR, "models")

if __name__ == "__main__":
    pass
