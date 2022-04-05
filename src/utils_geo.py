from __future__ import annotations, division, print_function

from itertools import product
from typing import List

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

from utils_paths import PATH_WORLD_BORDERS


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


def get_country_shape(iso2: str):
    """
    Explode muti-part geometries into multiple single geometries.

    Each row containing a multi-part geometry will be split into multiple rows with single geometries, thereby increasing the vertical size of the GeoDataFrame.
    """
    world_shape: gpd.GeoDataFrame = gpd.read_file(str(PATH_WORLD_BORDERS))
    country_shape = world_shape[world_shape["ISO2"] == iso2]
    country_shape = country_shape.explode()
    return country_shape


def get_intersecting_polygons(grid: List[Polygon], base_shape: List[Polygon], percentage_of_intersection_threshold=0):
    """
    Args:
        grid - list of polygons that will be checked against base_shape. polygons in grid which intersect with base_shape will be returned

        percentage_of_intersection_threshold:
            0 - find all intersections no matter how small
            0.3 - find all intersections where intersection area is atleast 30% compated to grid's polygon
            1 - find intersections where grid's polygon is fully inside of any base's polygons
    """

    intersecting_polygons = []
    for polygon_grid, polygon_base in product(grid, base_shape):
        is_area_valid = (polygon_grid.intersection(polygon_base).area / polygon_grid.area) > percentage_of_intersection_threshold
        if is_area_valid and polygon_grid not in intersecting_polygons:
            intersecting_polygons.append(polygon_grid)
    print(len(intersecting_polygons))
    return intersecting_polygons
