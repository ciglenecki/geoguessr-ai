""" Map the predictions to the weighted centroid if the prediction is outside of bounds of the country """
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from shapely.ops import nearest_points

from defaults import DEFAULT_COUNTRY_ISO2, DEFAULT_CROATIA_CRS, DEFAULT_GLOBAL_CRS, DEFAULT_SPACING
from preprocess_sample_coords import reproject_dataframe
from utils_functions import get_timestamp, is_valid_dir
from utils_geo import (
    ClippedCentroid,
    get_clipped_centroids,
    get_country_shape,
    get_grid,
    get_intersecting_polygons,
    minimal_distance_from_point_to_geodataframe,
)
from utils_paths import PATH_FIGURE, PATH_WORLD_BORDERS
from itertools import product
from typing import List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.point import Point
from shapely.ops import nearest_points
from tqdm import tqdm
import pandas as pd


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv-runtime",
        required=True,
        help="CSV that was created by the train.py in the runtime. It contains additional columns like lat_weighted, crs_x_weighted...",
    )

    parser.add_argument(
        "--csv-predictions",
        required=True,
        help="Predictions created either by the server or the predict.py",
    )
    parser.add_argument(
        "--out",
        help="Path to the csv output",
    )

    args = parser.parse_args()
    return args


def main(args, df_object=None):
    args = parse_args(args)

    path_out = args.out
    if args.out:
        if Path(path_out).stem == Path(path_out).name:
            path_out = Path(str(path_out) + ".csv")
    else:
        path_out = Path(
            Path(args.csv_predictions).parent,
            "{}_mapped_to_centroids.csv".format(Path(args.csv_predictions).name),
        )
    default_crs = DEFAULT_GLOBAL_CRS

    df_predict = pd.read_csv(Path(args.csv_predictions), index_col=False)
    df_regions = pd.read_csv(Path(args.csv_runtime), index_col=False)

    df_regions = df_regions.loc[:, ["lat_weighted", "lng_weighted"]].drop_duplicates()
    df_regions.rename(columns={"lat_weighted": "latitude", "lng_weighted": "longitude"}, inplace=True)

    df_predict_geo = gpd.GeoDataFrame(
        df_predict, geometry=gpd.points_from_xy(df_predict.loc[:, "longitude"], df_predict.loc[:, "latitude"])
    )

    df_regions_geo = gpd.GeoDataFrame(
        df_regions, geometry=gpd.points_from_xy(df_regions.loc[:, "longitude"], df_regions.loc[:, "latitude"])
    )

    print(df_predict_geo.head(n=3))
    print(df_regions_geo.head(n=3))

    world_shape: gpd.GeoDataFrame = gpd.read_file(str(PATH_WORLD_BORDERS))
    country_shape = get_country_shape(world_shape, DEFAULT_COUNTRY_ISO2)
    country_shape.set_crs(default_crs)

    new_df_dict = {"uuid": [], "latitude": [], "longitude": []}
    for _, row in tqdm(df_predict_geo.iterrows()):

        uuid, lat, lng = row["uuid"], row["latitude"], row["longitude"]
        new_df_dict["uuid"].append(uuid)

        if country_shape.contains(row.geometry).any():
            new_df_dict["latitude"].append(lat)
            new_df_dict["longitude"].append(lng)
        else:
            new_point = minimal_distance_from_point_to_geodataframe(Point(lng, lat), df_regions_geo)
            new_df_dict["latitude"].append(new_point.y)
            new_df_dict["longitude"].append(new_point.x)

    output_df = pd.DataFrame.from_dict(new_df_dict)
    output_df.to_csv(path_out, index=False)

    print("Saved file to:", path_out)


if __name__ == "__main__":
    main(sys.argv[1:])
