import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from defaults import DEFAULT_SPACING
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
        help="Dataframe you want to enrich",
    )

    parser.add_argument(
        "--out",
        metavar="dir",
        type=is_valid_dir,
        help="Directory where the enriched dataframe will be saved",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        help="Spacing that will be used to create a grid of polygons.",
        default=DEFAULT_SPACING,
    )

    parser.add_argument(
        "--out-fig",
        metavar="dir",
        type=is_valid_dir,
        help="Directory where the figure will be saved",
        default=PATH_FIGURE,
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


def append_polygons_without_data(df: pd.DataFrame, df_label_polygon_map: pd.DataFrame):
    """To the dataframe, append polygons for which the data (images) doesn't exist. Image related properties will be set to null"""
    df_labels_with_images = df["polygon_index"].dropna().unique()

    df_polygons_without_images = df_label_polygon_map.drop(df_labels_with_images)
    df = pd.concat([df, df_polygons_without_images])
    return df


def _handle_arguments(args, df_object):
    path_csv = args.csv
    no_out = args.no_out
    out = args.out
    out_dir_csv = out

    if df_object is None and path_csv is None:
        raise argparse.ArgumentError(
            args,
            "Provide one of the following: --csv path.csv or df_object via main function",
        )

    if not no_out:
        if out is None:
            if path_csv is None:
                raise argparse.ArgumentError(
                    args,
                    "You want to save the datarame but you didn't provide the output path",
                )
            out_dir_csv = Path(path_csv).parents[0]
    return path_csv, no_out, out_dir_csv


def generate_spherical_coords(
    latitude_column, longitude_column, centroid_lat_columns, centroid_lng_column
):

    x_coords = []
    y_coords = []
    z_coords = []
    x_centroid = []
    y_centroid = []
    z_centroid = []

    for lat, lng, c_lat, c_lng in zip(
        latitude_column, longitude_column, centroid_lat_columns, centroid_lng_column
    ):
        x_coords.append(math.sin(math.radians(lat)) * math.cos(math.radians(lng)))
        y_coords.append(math.sin(math.radians(lat)) * math.sin(math.radians(lng)))
        z_coords.append(math.cos(math.radians(lat)))
        x_centroid.append(math.sin(math.radians(c_lat)) * math.cos(math.radians(c_lng)))
        y_centroid.append(math.sin(math.radians(c_lat)) * math.sin(math.radians(c_lng)))
        z_centroid.append(math.cos(math.radians(c_lat)))

    return x_coords, y_coords, z_coords, x_centroid, y_centroid, z_centroid


def main(args, df_object=None):
    args = parse_args(args)
    path_csv, no_out, out_dir_csv = _handle_arguments(args, df_object)
    spacing, out_dir_fig, fig_format = args.spacing, args.out_fig, args.fig_format

    croatia_crs = 3766
    default_crs = 4326

    df = (
        df_object
        if type(df_object) is pd.DataFrame
        else pd.read_csv(path_csv, index_col=False)
    )
    df_geo_csv = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.loc[:, "longitude"], df.loc[:, "latitude"])
    )
    df.drop(
        "geometry", axis=1, inplace=True
    )  # info: GeoDataFrame somehow adds "geometry" column onto df

    world_shape: gpd.GeoDataFrame = gpd.read_file(str(PATH_WORLD_BORDERS))
    country_shape = get_country_shape(world_shape, "HR")
    country_shape.set_crs(default_crs)
    x_min, y_min, x_max, y_max = country_shape.total_bounds
    grid_polygons = get_grid(x_min, y_min, x_max, y_max, spacing=spacing)
    intersecting_polygons = get_intersecting_polygons(
        grid_polygons, country_shape.geometry, percentage_of_intersection_threshold=0
    )
    clipped_centroid = get_clipped_centroids(intersecting_polygons, country_shape)
    num_of_polygons = len(clipped_centroid)

    polygon_dict = {
        "polygon_index": [],
        "centroid_lat": [],
        "centroid_lng": [],
        "is_true_centroid": [],
    }
    polys_with_data = []

    for polygon_idx, (polygon, centroid) in tqdm(
        enumerate(zip(intersecting_polygons, clipped_centroid)),
        desc="Mapping image coords to polygon",
    ):

        centroid: ClippedCentroid
        row_mask = df_geo_csv.within(polygon)

        if row_mask.any():  # image existis inside this polygon
            df.loc[row_mask, "polygon_index"] = polygon_idx
            df.loc[row_mask, "centroid_lng"] = centroid.point.x
            df.loc[row_mask, "centroid_lat"] = centroid.point.y
            df.loc[row_mask, "is_true_centroid"] = centroid.is_true_centroid
            polys_with_data.append(polygon)

        polygon_dict["polygon_index"].append(polygon_idx)
        polygon_dict["centroid_lng"].append(centroid.point.x)
        polygon_dict["centroid_lat"].append(centroid.point.y)
        polygon_dict["is_true_centroid"].append(centroid.is_true_centroid)

    df_label_polygon_map = pd.DataFrame.from_dict(polygon_dict)

    # df['cart_x'], df['cart_y'], df['cart_z'], df['centroid_x'], df['centroid_y'], df['centroid_z'] = generate_spherical_coords(df['latitude'], df['longitude'], df['centroid_lat'], df['centroid_lng'])

    # country_shape = country_shape.to_crs(crs=croatia_crs)
    points_geometry = gpd.points_from_xy(df.loc[:, "longitude"], df.loc[:, "latitude"])
    df_csv = gpd.GeoDataFrame(df, geometry=points_geometry, crs=default_crs)  # type: ignore #[geopandas doesnt recognize args]
    df = reproject_dataframe(df_csv, croatia_crs)
    df = df.drop(["geometry"], axis=1)
    df = append_polygons_without_data(df, df_label_polygon_map)
    num_polygons_without_images = len(df.loc[df["uuid"].isna(), :])
    polys_without_data = [
        poly for poly in intersecting_polygons if poly not in polys_with_data
    ]

    print(
        "{}/{} images got marked by a polygon label (class)".format(
            df["polygon_index"].notnull().sum(), len(df)
        )
    )
    print(
        "{}/{} polygons have at least one image assigned to them".format(
            num_of_polygons - num_polygons_without_images, num_of_polygons
        )
    )
    num_valid_polys = len(intersecting_polygons)
    print(num_valid_polys, "- number of classes (polygons)")

    if no_out:
        return df

    filepath_csv = Path(
        out_dir_csv,
        "data__spacing_{}__num_class_{}.csv".format(spacing, num_valid_polys),
    )
    df.to_csv(filepath_csv, index=False)
    print("Saved file:", filepath_csv)

    filepath_csv_map = Path(
        out_dir_csv,
        "label_map__spacing_{}__num_class_{}.csv".format(spacing, num_valid_polys),
    )
    df_label_polygon_map.to_csv(filepath_csv_map, index=False)

    df_polys_with_data = gpd.GeoDataFrame({"geometry": polys_with_data})
    df_polys_without_data = gpd.GeoDataFrame({"geometry": polys_without_data})
    intersecting_points_df = gpd.GeoDataFrame(
        {"geometry": [c.point for c in clipped_centroid]}
    )

    ax = country_shape.plot(color="green")

    df_polys_with_data.plot(
        ax=ax, alpha=1, facecolor="none", linewidth=0.6, edgecolor="black"
    )
    df_polys_without_data.plot(
        ax=ax, alpha=1, facecolor="none", linewidth=0.6, edgecolor="red"
    )

    intersecting_points_df.plot(
        ax=ax, alpha=1, linewidth=0.2, markersize=2, edgecolor="white", color="red"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    filepath_fig = Path(
        out_dir_fig,
        "data_fig__spacing_{}__num_class_{}.{}".format(
            spacing, num_valid_polys, fig_format
        ),
    )

    plt.savefig(filepath_fig, dpi=1200)
    print("Saved file:", filepath_fig)
    return df


if __name__ == "__main__":
    main(sys.argv[1:])
