import argparse
import os
from pathlib import Path
from utils_functions import is_valid_dir
from utils_geo import ClippedCentroid, get_clipped_centroids, get_country_shape, get_grid, get_intersecting_polygons
import geopandas as gpd
import pandas as pd
from utils_paths import PATH_DATA_RAW, PATH_FIGURE
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: project to CRS (?) then caculate distances, then re-project (?)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path(PATH_DATA_RAW, "data.csv")),
        type=str,
        help="Path to dataframe you want to enrich",
    )

    parser.add_argument(
        "--out",
        metavar="dir",
        type=is_valid_dir,
        help="Directory where the enriched dataframe will be saved",
        required=False,
    )

    parser.add_argument(
        "--spacing",
        type=float,
        help="Spacing that will be used to create a grid of polygons.",
        default=0.2,
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    path_csv = args.csv
    out_dir_csv = args.out if args.out else os.path.dirname(args.csv)
    spacing = args.spacing
    out_dir_fig = args.out_fig
    fig_format = args.fig_format

    df = pd.read_csv(path_csv, index_col=False)
    df_geo_csv = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.loc[:, "longitude"], df.loc[:, "latitude"]))
    df.drop("geometry", axis=1, inplace=True)  # info: GeoDataFrame(df ... somehow adds "geometry" onto df

    overlap_threshold = 0

    country_shape = get_country_shape("HR")
    x_min, y_min, x_max, y_max = country_shape.total_bounds
    grid_polygons = get_grid(x_min, y_min, x_max, y_max, spacing=spacing)
    intersecting_polygons = get_intersecting_polygons(grid_polygons, country_shape.geometry, percentage_of_intersection_threshold=overlap_threshold)
    clipped_centroid = get_clipped_centroids(intersecting_polygons, country_shape)

    for i, (polygon, centroid) in tqdm(enumerate(zip(intersecting_polygons, clipped_centroid)), desc="Mapping image coords to polygon"):
        centroid: ClippedCentroid
        row_mask = df_geo_csv.within(polygon)
        df.loc[row_mask, "label"] = i
        df.loc[row_mask, "centroid_lat"] = centroid.point.x
        df.loc[row_mask, "centroid_lng"] = centroid.point.y
        df.loc[row_mask, "is_true_centroid"] = centroid.is_true_centroid

    print("{}/{} images got marked by a polygon label (class)".format(df["label"].notnull().sum(), len(df)))
    df = df.loc[~df["label"].isnull(), :]

    num_classes = len(intersecting_polygons)
    print(num_classes, "- number of classes (polygons)")

    filepath_csv = Path(out_dir_csv, "data__num_class_{}__spacing_{}.csv".format(num_classes, spacing))
    df.to_csv(filepath_csv, index=False)
    print("Saved file:", filepath_csv)

    intersecting_polygons_df = gpd.GeoDataFrame({"geometry": intersecting_polygons})
    intersecting_points_df = gpd.GeoDataFrame({"geometry": [c.point for c in clipped_centroid]})
    ax = country_shape.plot(color="green")
    intersecting_polygons_df.plot(ax=ax, alpha=0.8, facecolor="none", linewidth=0.8, edgecolor="black")
    intersecting_points_df.plot(ax=ax, alpha=1, linewidth=0.2, markersize=2, edgecolor="white", color="red")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    filepath_fig = Path(out_dir_fig, "data_fig__num_class_{}__spacing_{}.{}".format(num_classes, spacing, fig_format))

    plt.savefig(filepath_fig, dpi=1200)
    print("Saved file:", filepath_fig)
