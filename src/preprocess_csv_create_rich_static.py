import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Union, cast

import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from config import cfg
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--csv", help="Path the base csv you want to enrich", required=True
    )

    parser.add_argument(
        "--hex-side",
        type=float,
        help="""Class area in km^2.
                0.7 spacing => ~31 classes
                0.5 spacing => ~55 classes
                0.4 spacing => ~75 classes
                0.3 spacing => ~115 classes
        """,
        default=10_000,
    )

    parser.add_argument(
        "--out-dir-fig",
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
        "--out",
        metavar="dir",
        type=is_valid_dir,
        help="Directory where the enriched dataframe will be saved",
    )
    parser.add_argument(
        "--global-crs",
        type=int,
        default=cfg.geo.global_crs,
    )
    parser.add_argument(
        "--local-crs",
        type=int,
        default=cfg.geo.area.local_crs,
    )
    parser.add_argument(
        "--country-iso2",
        metavar="name",
        type=str,
        default=cfg.geo.area.country_iso2,
    )
    args = parser.parse_args(args)
    args = _handle_arguments(args)
    print(args)
    return args


def save_fig(path: Union[str, Path]):
    plt.savefig(path, dpi=300)
    print("Saved figure: {}".format(path))


def plot_country(country_shape: gpd.GeoDataFrame, ax_arg=None):

    ax = country_shape.plot(color="green", ax=ax_arg)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.axis("scaled")
    return ax


def append_polygons_without_images(
    df: pd.DataFrame, polygon_dict: Dict[str, Dict[str, Any]]
):
    """To the dataframe, append polygons for which the data (images) doesn't exist. Image related properties will be set to null"""

    new_dict = [
        {"polygon_index": key, **value}
        for key, value in polygon_dict.items()
        if not value["has_image"]
    ]
    df_tmp = pd.DataFrame.from_records(new_dict)
    df_tmp = df_tmp.loc[df_tmp["has_image"] == False]
    df_tmp = df_tmp.drop("has_image", axis=1)
    df = pd.concat([df, df_tmp])
    return df


def _handle_arguments(args):
    if not args.out:
        args.out = Path(args.csv).parents[0]
    args.out = Path(args.out) if args.out else Path(args.csv).parents[0]
    return args


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


def generate_src_coords(lat: pd.Series, lng: pd.Series):
    points_geometry = gpd.points_from_xy(lng, lat)  # x is lng y is lat
    df_tmp = gpd.GeoDataFrame(
        columns=["x", "y"], geometry=points_geometry, crs=cfg.geo.area.local_crs
    )  # type ignore #[geopandas doesnt recognize args]
    df_tmp: gpd.GeoDataFrame = df_tmp.to_crs(
        cfg.geo.area.local_crs
    )  # type ignore it cant distinguish from geo/pandas

    def handle_dot(point: Point, x_or_y: str):
        if point.is_empty:
            return None
        return point.x if x_or_y == "x" else point.y

    df_tmp["x"] = df_tmp.geometry.apply(lambda p: handle_dot(p, "x"))
    df_tmp["y"] = df_tmp.geometry.apply(lambda p: handle_dot(p, "y"))
    return df_tmp["y"], df_tmp["x"]


def replace_values_with_na_invalid_locations(df: pd.DataFrame):
    mask_rows_with_invalid_projected_coords = df.isnull().any(axis=1)

    print(
        "\nThe following locations with invalid projections will be excluded:",
        df.loc[mask_rows_with_invalid_projected_coords, "uuid"].to_list(),
    )

    df.loc[
        mask_rows_with_invalid_projected_coords,
        ~df.columns.isin(["uuid", "latitude", "longitude"]),
    ] = None
    return df


# def _build_init_dataframe(csv: Path | str):

#     return df


def _get_country_shape(
    shp_file: str | Path, country_iso2: str, global_crs: int, local_crs: int
) -> gpd.GeoDataFrame:
    world_shape = cast(gpd.GeoDataFrame, gpd.read_file(shp_file))
    world_shape.set_crs(global_crs)
    country_shape = get_country_shape(world_shape, country_iso2)
    country_df = cast(gpd.GeoDataFrame, country_shape.to_crs(local_crs))
    return country_df


def main(args, df_object=None) -> Union[str, pd.DataFrame]:
    args = parse_args(args)
    df = pd.read_csv(args.csv, index_col=False)
    df_geo_csv = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df.loc[:, "longitude"], df.loc[:, "latitude"], crs=args.global_crs
        ),
    )  # type ignore
    df.drop(
        "geometry", axis=1, inplace=True
    )  # info: GeoDataFrame somehow adds "geometry" column onto the df

    country_shape = _get_country_shape(
        PATH_WORLD_BORDERS, args.country_iso2, args.global_crs, args.local_crs
    )
    x_min, y_min, x_max, y_max = country_shape.total_bounds
    grid_polygons = get_grid(x_min, y_min, x_max, y_max, hexagon_side=args.hex_side)
    print(grid_polygons)
    intersecting_polygons = get_intersecting_polygons(
        grid_polygons, country_shape.geometry, percentage_of_intersection_threshold=0
    )
    clipped_centroid = get_clipped_centroids(intersecting_polygons, country_shape)
    num_of_polygons = len(clipped_centroid)

    polys_all_dict: Dict[str, Dict[str, Any]] = {}
    polys_with_images: Dict[str, Polygon] = {}
    polys_without_images: Dict[str, Polygon] = {}

    for polygon_idx, (polygon, centroid) in tqdm(
        enumerate(zip(intersecting_polygons, clipped_centroid)),
        desc="Mapping image coords to polygon",
    ):

        centroid: ClippedCentroid
        row_mask = df_geo_csv.within(polygon)

        poly_has_images = row_mask.any()

        if poly_has_images:  # image existis inside this polygon
            df.loc[row_mask, "polygon_index"] = polygon_idx
            df.loc[row_mask, "centroid_lng"] = centroid.point.x
            df.loc[row_mask, "centroid_lat"] = centroid.point.y
            df.loc[row_mask, "is_true_centroid"] = centroid.is_true_centroid
            polys_with_images[polygon_idx] = polygon
        else:
            polys_without_images[polygon_idx] = polygon

        if polygon_idx not in polys_all_dict:
            polys_all_dict[polygon_idx] = {}
        polys_all_dict[polygon_idx]["centroid_lng"] = centroid.point.x
        polys_all_dict[polygon_idx]["centroid_lat"] = centroid.point.y
        polys_all_dict[polygon_idx]["is_true_centroid"] = centroid.is_true_centroid
        polys_all_dict[polygon_idx]["has_image"] = poly_has_images

    # df_label_polygon_map = pd.DataFrame.from_dict(polys_all_dict)
    print(polys_with_images, polys_without_images)
    print(df)
    exit(1)
    df["crs_y"], df["crs_x"] = generate_src_coords(
        df.loc[:, "latitude"], df.loc[:, "longitude"]
    )
    df["crs_centroid_y"], df["crs_centroid_x"] = generate_src_coords(
        df.loc[:, "centroid_lat"], df.loc[:, "centroid_lng"]
    )

    df = replace_values_with_na_invalid_locations(df)
    df = append_polygons_without_images(df, polys_all_dict)
    num_polygons_without_images = len(df.loc[df["uuid"].isna(), :])

    print(
        "\n{}/{} locations got have region assigned to them. Locations that do not belong to any region will later be excluded.".format(
            df["polygon_index"].notnull().sum(), len(df)
        )
    )
    print(
        "\n{}/{} regions have at least one image assigned to them".format(
            num_of_polygons - num_polygons_without_images, num_of_polygons
        )
    )
    num_valid_polys = len(intersecting_polygons)

    if no_out:
        return df

    path_csv_out = str(
        Path(
            out_csv,
            "{}_rich_static__spacing_{}_classes_{}.csv".format(
                Path(input_csv).stem, spacing, num_valid_polys
            ),
        )
    )

    pd.set_option("display.max_rows", 1000)
    print("\nNumber of images for each region\n", df.groupby(["polygon_index"]).size())
    pd.set_option("display.max_rows", None)

    print(
        "Mean of number of images in regions :\n",
        df.groupby(["polygon_index"]).size().mean(),
    )
    df.to_csv(path_csv_out, index=False)

    df_grid = gpd.GeoDataFrame({"geometry": grid_polygons})
    df_intersect_grid = gpd.GeoDataFrame({"geometry": intersecting_polygons})
    df_polys_with_images = gpd.GeoDataFrame({"geometry": polys_with_images})
    df_polys_without_images = gpd.GeoDataFrame({"geometry": polys_without_images})
    intersecting_points_df = gpd.GeoDataFrame(
        {"geometry": [c.point for c in clipped_centroid]}
    )
    basename = "data_fig__spacing_{}__num_class_{}".format(spacing, num_valid_polys)
    os.makedirs(out_dir_fig, exist_ok=True)

    ax = plot_country(country_shape)
    plt.xlim(xmax=20.25)
    plt.ylim(ymin=42)
    save_fig(Path(out_dir_fig, "country_{}_1.png".format(DEFAULT_COUNTRY_ISO2)))

    df_grid.plot(ax=ax, alpha=1, facecolor="none", linewidth=0.6, edgecolor="black")
    plt.xlim(xmax=20.25)
    plt.ylim(ymin=42)
    save_fig(
        Path(
            out_dir_fig,
            "country_{}_spacing_{}_grid_2.png".format(
                DEFAULT_COUNTRY_ISO2,
                spacing,
            ),
        )
    )

    plt.close()
    ax = plot_country(country_shape)
    plt.xlim(xmax=20.25)
    plt.ylim(ymin=42)
    df_intersect_grid.plot(
        ax=ax, alpha=1, facecolor="none", linewidth=0.6, edgecolor="black"
    )
    save_fig(
        Path(
            out_dir_fig,
            "country_{}_spacing_{}_grid_intersection_3.png".format(
                DEFAULT_COUNTRY_ISO2, spacing
            ),
        )
    )

    """ Plotting polygon index"""
    df_polys_with_images["coords"] = df_polys_with_images["geometry"].apply(
        lambda poly: (poly.centroid.x, poly.centroid.y)
    )
    df_polys_with_images.loc[
        :, "polygon_index"
    ] = polys_with_images.keys()  # type ignore
    for idx, row in df_polys_with_images.iterrows():
        ann = plt.annotate(
            text=row["polygon_index"],
            xy=row["coords"],
            xytext=(-24 * spacing, 24 * spacing),
            textcoords="offset points",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=5,
            color="white",
        )  # type ignore
        ann.set_path_effects(
            [
                path_effects.Stroke(linewidth=0.5, foreground="black"),
                path_effects.Normal(),
            ]
        )

    df_polys_without_images.plot(
        ax=ax, alpha=1, facecolor="none", linewidth=0.6, edgecolor="red"
    )
    df_polys_with_images.plot(
        ax=ax, alpha=1, facecolor="none", linewidth=0.6, edgecolor="black"
    )  # replot black squares
    plt.xlim(xmax=20.25)
    plt.ylim(ymin=42)
    save_fig(Path(out_dir_fig, basename + "_regions_colored_4" + "." + fig_format))

    intersecting_points_df.plot(
        ax=ax, alpha=1, linewidth=0.3, markersize=5, edgecolor="white", color="red"
    )
    save_fig(Path(out_dir_fig, basename + "centroid_5" + fig_format))

    print("\nRich static CSV saved: '{}'".format(path_csv_out))

    return path_csv_out


if __name__ == "__main__":
    main(sys.argv[1:])
