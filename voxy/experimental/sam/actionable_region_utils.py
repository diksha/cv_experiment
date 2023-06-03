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
"""
Given a camera UUID and an actor's bounding box, return whether actor is inside
the actionable region.
"""

import numpy as np
import cv2 as cv

from shapely.geometry import Polygon
from core.incidents.utils import CameraConfig


def get_actionable_region_polygon(camera_uuid):
    """Return actionable region(s) polygon given camera UUID"""
    return CameraConfig(camera_uuid).actionable_regions


def generate_masked_img(frame, actionable_region, output_path):
    """Take frame and actionable region and generate masked frame"""
    height, width, channels = frame.shape
    mask = np.zeros([height, width])
    mask.fill(1)  # create mask image and fill white
    points = np.array([[p.x, p.y] for p in actionable_region.vertices], np.int32)
    points = points.reshape((-1, 1, 2))
    mask = cv.fillPoly(mask, [points], 0)  # draw black polygon of region on mask
    mask = 1 - mask  # invert mask
    processed_frame = frame.copy()
    processed_frame[mask == 0] = 0
    processed_frame[mask != 0] = frame[mask != 0]  # mask out image
    if output_path != "":
        cv.imwrite(output_path, processed_frame)

    return processed_frame


def bbox_actionable_region_overlap(camera_uuid, bbox):
    """Given a bounding box, check if it overlaps with the actionable region
    for a certain camera"""
    bbox_poly = Polygon([[p.x, p.y] for p in bbox.vertices])
    actionable_polygon = get_actionable_region_polygon(camera_uuid)
    actionable_poly = Polygon([[p.x, p.y] for p in actionable_polygon.vertices])

    return bbox_poly.overlaps(actionable_poly)


def actionable_region_contains_bbox(camera_uuid, bbox):
    """Given a bounding box, check if it is within the actionable region for a
    certain camera"""
    bbox_poly = Polygon([[p.x, p.y] for p in bbox.vertices])
    actionable_polygon = get_actionable_region_polygon(camera_uuid)
    actionable_poly = Polygon([[p.x, p.y] for p in actionable_polygon.vertices])

    return actionable_poly.contains(bbox_poly)
