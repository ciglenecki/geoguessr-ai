from __future__ import annotations, division, print_function

from itertools import product
from typing import List

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box
from shapely.geometry.point import Point
from shapely.ops import nearest_points
from tqdm import tqdm

from utils_paths import PATH_WORLD_BORDERS


class ClippedCentroid:
    point: Point
    is_true_centroid: bool


def get_grid(x_min, y_min, x_max, y_max, spacing):
    """
    Args:
        x_min, y_min, x_max, y_max: bounds that form a grid
        spacing: width/height of each polygon

    Return:
        list of polygons that form a grid
    """
    polygons: List[Polygon] = []
    for y in np.arange(y_min, y_max + spacing, spacing):
        for x in np.arange(x_min, x_max + spacing, spacing):
            polygon = box(x, y, x + spacing, y + spacing)
            polygons.append(polygon)
    return polygons


def get_country_shape(world_shape: gpd.GeoDataFrame, iso2: str) -> gpd.GeoDataFrame:
    """
    Country shape dataframe has multi-part geometry (multi-polygons) as rows. By calling explode, multi-part geometry will be flattened. Dataframe will have more rows.
    """

    country_shape = world_shape[world_shape["ISO2"] == iso2]
    country_shape = country_shape.explode(ignore_index=False)
    country_shape = country_shape.droplevel(0)  # we don't need country index on level 0, we work with a single country
    return country_shape  # type: ignore [can't recognize type because of modifications]


def get_intersecting_polygons(grid: List[Polygon], base_shape: List[Polygon], percentage_of_intersection_threshold=0):
    """
    Args:
        grid - list of polygons that will be checked against base_shape. polygons in grid which intersect with base_shape will be returned

        percentage_of_intersection_threshold:
            0 - find all intersections no matter how small
            0.3 - find all intersections where intersection area is atleast 30% compated to grid's polygon
            1 - find intersections where grid's polygon is fully inside of any base's polygons
    """

    intersecting_polygons: List[Polygon] = []
    for polygon_grid, polygon_base in tqdm(product(grid, base_shape), desc="Finding polygons that intersect"):
        is_area_valid = (polygon_grid.intersection(polygon_base).area / polygon_grid.area) > percentage_of_intersection_threshold
        if is_area_valid and polygon_grid not in intersecting_polygons:
            intersecting_polygons.append(polygon_grid)
    return intersecting_polygons


def get_clipped_centroids(polygons: List[Polygon], clipping_shape: gpd.GeoDataFrame):
    """
    If polygon's centroid in not inside of clipping_shape, the appended value will be the closest point from the true centroid to the clipping_shape

    Args:
        polygons - polygons whose centroid will be caculated
        clipping_shape - shape that is used for clipping

    Example:
        clipping_shape is shape of the Croatia and polygon in a square that contains Island of Krk but it's centroid is somewhere in the sea...
        clipped centroid will be the closest point from the true centroid to the land of the Island of Krk
    """

    polygon_clipped_centroids: List[ClippedCentroid] = []
    for polygon in tqdm(polygons, desc="Finding clipped centroids of polygons"):
        clipped_centroid = ClippedCentroid()
        centroid = polygon.centroid
        if clipping_shape.contains(centroid).any():
            clipped_centroid.point = centroid
            clipped_centroid.is_true_centroid = True
        else:
            distances = clipping_shape.distance(centroid)
            country_polygon_index = distances.sort_values().index[0]
            country_polygon = clipping_shape.loc[country_polygon_index, :]
            nearest_point = nearest_points(centroid, country_polygon.geometry)[1]  # [0] is the first argument, [1] is nearest point

            clipped_centroid.point = nearest_point
            clipped_centroid.is_true_centroid = False
        polygon_clipped_centroids.append(clipped_centroid)
    return polygon_clipped_centroids
