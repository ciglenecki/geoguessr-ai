import argparse
from pathlib import Path
import pdb
import matplotlib.pyplot as plt
from utils_paths import PATH_DATA, PATH_DATA_RAW, PATH_WORLD_BORDERS
import pandas as pd
import shapefile
import matplotlib.patheffects as path_effects
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, box, MultiPoint, Point
import numpy as np


# parser = argparse.ArgumentParser()
# parser.add_argument("--driver_num", metavar="N", type=int, help="Number of drivers that will be used (1 >= N <= 12)")
# args = parser.parse_args()

if __name__ == "__main__":

    """Country grid generate"""

    def get_geometries(x_min, y_min, x_max, y_max, spacing=0.06):
        polygons = []
        points = []

        for y in np.arange(y_min, y_max + spacing, spacing):
            for x in np.arange(x_min, x_max + spacing, spacing):

                polygon = box(x, y, x + spacing, y + spacing)
                polygons.append(polygon)
                point = Point(x, y)
                points.append(point)

        return polygons, points

    world_shape = gpd.read_file(str(PATH_WORLD_BORDERS))
    country_shape = world_shape[world_shape["ISO2"] == "HR"]
    x_min, y_min, x_max, y_max = country_shape.total_bounds

    polygons, points = get_geometries(x_min, y_min, x_max, y_max)
    grid = gpd.GeoDataFrame({"geometry": polygons})
    grid_points_df = gpd.GeoDataFrame({"geometry": polygons})

    intersecting_polygons = []
    intersecting_points = []
    # print(country_shape.geometry.any())
    for polygon in grid.geometry:
        for country_s in country_shape.geometry.any():
            if polygon.intersects(country_s):
                if (polygon.intersection(country_s).area / polygon.area) >= 0.3:
                    intersecting_polygons.append(polygon)

    for point in grid_points_df.geometry:
        if country_shape.geometry.any().contains(point):
            intersecting_points.append(point)

    print(len(intersecting_points))
    print(len(intersecting_polygons))
    intersecting_polygons_df = gpd.GeoDataFrame({"geometry": intersecting_polygons})
    intersecting_points_df = gpd.GeoDataFrame({"geometry": intersecting_points})

    base = country_shape.plot(color="green")
    intersecting_polygons_df.plot(ax=base, alpha=0.3, linewidth=0.2, edgecolor="black")
    intersecting_points_df.plot(ax=base, alpha=0.3, linewidth=0.2, edgecolor="black")
    plt.show()

    # for i, g1 in enumerate(geom_p1):
    #     for j, g8 in enumerate(geom_p8):
    #         if g1.intersects(g8):
    #             print i, j, (g1.intersection(g8).area/g1.area)*100
