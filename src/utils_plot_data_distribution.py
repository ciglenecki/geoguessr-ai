import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from config import (
    DEFAULT_COUNTRY_ISO2,
    DEFAULT_CROATIA_CRS,
    DEFAULT_GLOBAL_CRS,
    DEFAULT_SPACING,
)
from preprocess_sample_coords import reproject_dataframe
from utils_functions import is_valid_dir
from utils_geo import (
    ClippedCentroid,
    get_clipped_centroids,
    get_country_shape,
    get_grid,
    get_intersecting_polygons,
)
from utils_paths import PATH_FIGURE, PATH_WORLD_BORDERS


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        help="Dataframe with data",
    )

    parser.add_argument(
        "--fig-format",
        type=str,
        choices=list(plt.gcf().canvas.get_supported_filetypes().keys()),
        help="Supported file formats for matplotlib savefig",
        default="png",
    )

    parser.add_argument(
        "--no-out",
        action="store_true",
        help="Disable any dataframe or figure saving. Useful when calling inside other scripts",
    )

    args = parser.parse_args(args)
    return args


def main(args, df_object=None):
    args = parse_args(args)
    out_dir_fig, fig_format = args.out_fig, args.fig_format
    path_csv = args.csv

    fig_format = "." + fig_format

    df = (
        df_object
        if type(df_object) is pd.DataFrame
        else pd.read_csv(path_csv, index_col=False)
    )
    world_shape: gpd.GeoDataFrame = gpd.read_file(str(PATH_WORLD_BORDERS))
    country_shape = get_country_shape(world_shape, DEFAULT_COUNTRY_ISO2)
    df_geo_csv = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.loc[:, "longitude"], df.loc[:, "latitude"])
    )

    ax = country_shape.plot(color="green")
    df_geo_csv.plot(
        ax=ax, alpha=1, linewidth=0.2, markersize=1, edgecolor="white", color="red"
    )
    file_path = Path(out_dir_fig, "croatia_data_distribution.png")
    if args.no_out:
        plt.plot()
        return
    plt.savefig(file_path, dpi=1200)
    print("Saved to file", str(file_path))


if __name__ == "__main__":
    main(sys.argv[1:])
