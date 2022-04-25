import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_functions import is_valid_dir
from utils_geo import get_country_shape
from utils_paths import PATH_DATA_SAMPLER, PATH_WORLD_BORDERS

np.random.seed(0)


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        metavar="dir",
        type=is_valid_dir,
        help="Directory where the dataframe will be saved",
        default=PATH_DATA_SAMPLER,
    )

    parser.add_argument(
        "--n",
        type=int,
        help="Number of sampled coordinates",
        default=1_000_000,
    )

    parser.add_argument(
        "--fig-format",
        type=str,
        choices=list(plt.gcf().canvas.get_supported_filetypes().keys()),
        help="Supported file formats for matplotlib savefig",
        default="png",
    )

    parser.add_argument(
        "--no-fig-out",
        action="store_true",
        help="Disable figure saving.",
    )

    parser.add_argument(
        "--no-df-out",
        action="store_true",
        help="Disable dataframe saving.",
    )

    args = parser.parse_args(args)
    return args


def uniform_2d_generator(lat_lng_min, lat_lng_max, batch_size):
    while True:
        yield np.random.uniform(low=lat_lng_min, high=lat_lng_max, size=(batch_size, 2))


def reproject_dataframe(df, crs):
    print("Reprojecting the dataframe...")
    df = df.to_crs(crs)
    df["sample_longitude"] = df.geometry.apply(lambda p: p.x)
    df["sample_latitude"] = df.geometry.apply(lambda p: p.y)
    return df


def main(args):
    args = parse_args(args)
    out_dir = args.out
    no_fig_out = args.no_fig_out
    no_df_out = args.no_df_out
    num_of_coords = args.n
    fig_format = args.fig_format

    croatia_crs = 3766
    default_crs = 4326

    batch_size = num_of_coords // 5

    world_shape: gpd.GeoDataFrame = gpd.read_file(str(PATH_WORLD_BORDERS))
    country_shape = get_country_shape(world_shape, "HR")
    country_shape.set_crs(default_crs)
    country_shape = country_shape.to_crs(crs=croatia_crs)
    if country_shape is None:
        raise Exception("country_shape is none")

    lng_min, lat_min, lng_max, lat_max = country_shape.total_bounds
    lat_lng_min = [lat_min, lng_min]
    lat_lng_max = [lat_max, lng_max]

    final_df = gpd.GeoDataFrame()
    for rand_coords in tqdm(uniform_2d_generator(lat_lng_min, lat_lng_max, batch_size)):
        df_batch = pd.DataFrame(rand_coords, columns=["latitude", "longitude"])
        points_geometry = gpd.points_from_xy(df_batch.loc[:, "longitude"], df_batch.loc[:, "latitude"])
        df_batch_csv = gpd.GeoDataFrame(df_batch, geometry=points_geometry, crs=croatia_crs)  # type: ignore #[geopandas doesnt recognize args]
        df_batch_points_in_country = gpd.sjoin(df_batch_csv, country_shape, predicate="within").set_crs(croatia_crs)
        final_df = gpd.GeoDataFrame(pd.concat([final_df, df_batch_points_in_country]), crs=croatia_crs)

        if len(final_df) >= num_of_coords:
            break

    country_shape = country_shape.to_crs(default_crs)
    if country_shape is None:
        raise Exception("country_shape is none")
    final_df = final_df.set_crs(croatia_crs)
    final_df = reproject_dataframe(final_df, default_crs)

    """ Saving files """
    basename = "coords_sample__n_{}".format(num_of_coords)
    if not no_df_out:
        print("Saving to csv...")
        final_df_clean = final_df.loc[:, ["sample_longitude", "sample_latitude"]]
        final_df_clean = final_df_clean.sample(frac=1).reset_index(drop=True)  # Shuffle rows and reassign indices
        csv_path = Path(out_dir, basename + ".csv")
        final_df_clean.to_csv(csv_path, index_label=False)
        print("Saved file", str(csv_path))

    if not no_fig_out:
        print("Saving figure...")
        ax = country_shape.plot(color="green")
        final_df.plot(ax=ax, alpha=1, linewidth=0.01, markersize=0.01, edgecolor="white", color="red")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig_path = Path(out_dir, basename + "." + fig_format)
        plt.title("Uniform sample of coordinates (n={})".format(num_of_coords))
        plt.savefig(fig_path, dpi=1200)
        print("Saved file", str(fig_path))
    return final_df


if __name__ == "__main__":
    main(sys.argv[1:])
