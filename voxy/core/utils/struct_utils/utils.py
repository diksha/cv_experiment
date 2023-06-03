#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from core.structs.attributes import Polygon


def get_shapely_polygon(polygon: Polygon) -> ShapelyPolygon:
    """Transforms struct polygon to shapely polygon

    Args:
        polygon (Polygon): polygon defining bounding boxes

    Returns:
        ShapelyPolygon: transformed to a shapely type.
    """
    return ShapelyPolygon([[p.x, p.y] for p in polygon.vertices])


def get_iou(polygon_a: Polygon, polygon_b: Polygon) -> float:
    """Intersection over union for two polygons."""
    shapely_polygon_a = get_shapely_polygon(polygon_a)
    shapely_polygon_b = get_shapely_polygon(polygon_b)

    return (
        shapely_polygon_a.intersection(shapely_polygon_b).area
        / shapely_polygon_a.union(shapely_polygon_b).area
    )


def get_pct_intersection(polygon_a: Polygon, polygon_b: Polygon) -> float:
    """Percentage Intersection over the area of polygon a.
    Eg. PIT (a) intersection DOOR (b) determines % of PIT inside the DOOR

    Args:
        polygon_a (Polygon): Bounding box of a
        polygon_b (Polygon): Bounding box of b

    Returns:
        float: ((a intersection b) / area of a) --> %
    """
    shapely_polygon_a = get_shapely_polygon(polygon_a)
    shapely_polygon_b = get_shapely_polygon(polygon_b)

    return (
        shapely_polygon_a.intersection(shapely_polygon_b).area
        / shapely_polygon_a.area
    )


def get_bbox_from_xysr(xysr_array: np.ndarray) -> np.ndarray:
    """
    Converts an array of xysr values (n x 4) to bbox values (N x 4)

    Args:
        xysr_array (np.ndarray): array of xysr values of shape n x 4

    Returns:
        bbox_array (np.ndarray): array of bbox values of shape n x 4
    """
    if xysr_array.ndim == 1:
        xysr_array = xysr_array.reshape(-1, xysr_array.size)
    x_array = xysr_array[:, 0]
    y_array = xysr_array[:, 1]
    s_array = xysr_array[:, 2]
    r_array = xysr_array[:, 3]
    w_array = np.sqrt(s_array / r_array)
    h_array = r_array * w_array
    x1 = x_array - w_array / 2.0
    x2 = x_array + w_array / 2.0
    y1 = y_array - h_array / 2.0
    y2 = y_array + h_array / 2.0
    return np.array([x1, y1, x2, y2]).T
