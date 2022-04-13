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
PATH_DATA_EXTERNAL = Path(PATH_DATA, "external.ignoreme")
PATH_DATA_RAW = Path(PATH_DATA, "raw.ignoreme")
# PATH_WORLD_BORDERS = Path(PATH_DATA, "world-borders/TM_WORLD_BORDERS-0.3.shp")
PATH_WORLD_BORDERS = Path(PATH_DATA, "world-borders", "world_no_croatia_islands.shp")
PATH_DATA_CSV_DECORATED = Path(PATH_DATA, "csv_decorated")

PATH_REPORT = Path(WORK_DIR, "reports")
PATH_FIGURE = Path(PATH_REPORT, "figures")
PATH_MODEL = Path(WORK_DIR, "models")

if __name__ == "__main__":
    pass
