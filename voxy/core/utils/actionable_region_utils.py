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
"""Utilities for working with actionable regions."""

import cv2 as cv
import numpy as np
from shapely.geometry import Polygon

from core.incidents.utils import CameraConfig
from core.structs.actor import Actor


def get_actionable_region_polygons(camera_uuid, frame_height, frame_width):
    """Return actionable region(s) polygon given camera UUID."""
    return CameraConfig(
        camera_uuid, frame_height, frame_width
    ).actionable_regions


def generate_masked_img(frame, actionable_region, output_path):
    """Take frame and actionable region and generate masked frame."""
    height, width, channels = frame.shape
    mask = np.zeros([height, width])
    mask.fill(1)  # create mask image and fill white
    points = np.array(
        [[p.x, p.y] for p in actionable_region.vertices], np.int32
    )
    points = points.reshape((-1, 1, 2))
    mask = cv.fillPoly(
        mask, [points], 0
    )  # draw black polygon of region on mask
    mask = 1 - mask  # invert mask
    processed_frame = frame.copy()
    processed_frame[mask == 0] = 0
    processed_frame[mask != 0] = frame[mask != 0]  # mask out image
    if output_path != "":
        cv.imwrite(output_path, processed_frame)

    return processed_frame


def person_inside_actionable_region(
    camera_uuid: str, person_actor: Actor, frame_height: int, frame_width: int
) -> bool:
    """Checks if the person is standing inside an actionable region

    Args:
        camera_uuid (str): uuid of the camera to fetch camera configs like actionable region from
        person_actor (Actor): Actor struct represention of the person actor
        frame_height (int): frame height of the camera feed for de-normalization
        frame_width (int): frame width of the camera feed for de-normalization

    Returns:
        bool: whether a person is inside an actionable region or not
    """
    actionable_polygons = get_actionable_region_polygons(
        camera_uuid, frame_height, frame_width
    )

    is_person_in_actionable_region = True
    person_floor_center = (
        person_actor.polygon.get_bottom_center().to_shapely_point()
    )
    for actionable_polygon in actionable_polygons:
        actionable_shapely_poly = Polygon(
            [[p.x, p.y] for p in actionable_polygon.vertices]
        )

        is_person_in_actionable_region = actionable_shapely_poly.contains(
            person_floor_center
        )
        if is_person_in_actionable_region:
            break
    return is_person_in_actionable_region
