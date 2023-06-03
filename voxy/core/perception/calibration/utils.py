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

from typing import Callable

import numpy as np
from shapely.geometry import Polygon

from core.perception.calibration.camera_model import CameraModel
from core.structs.extrinsics import Extrinsics
from core.structs.intrinsics import Intrinsics


def calibration_config_to_camera_model(config: dict) -> CameraModel:
    # intrinsics
    intrinsic_distortion = config["calibration"]["intrinsics"][
        "distortion_usm"
    ]
    intrinsic_focal_length = config["calibration"]["intrinsics"][
        "focal_length_pixels"
    ]
    intrinsic_width = config["calibration"]["intrinsics"]["width_pixels"]
    intrinsic_height = config["calibration"]["intrinsics"]["height_pixels"]
    intrinsic_cx = config["calibration"]["intrinsics"]["cx_pixels"]
    intrinsic_cy = config["calibration"]["intrinsics"]["cy_pixels"]

    # extrinsics
    extrinsic_z_m = config["calibration"]["extrinsics"]["z_m"]
    extrinsic_roll_rad = config["calibration"]["extrinsics"]["roll_radians"]
    extrinsic_pitch_rad = config["calibration"]["extrinsics"]["pitch_radians"]

    # models
    intrinsics = Intrinsics(
        intrinsic_focal_length,
        intrinsic_distortion,
        intrinsic_cx,
        intrinsic_cy,
        intrinsic_width,
        intrinsic_height,
    )
    extrinsics = Extrinsics(
        extrinsic_z_m, extrinsic_roll_rad, extrinsic_pitch_rad
    )
    camera_model = CameraModel(intrinsics, extrinsics)
    return camera_model


def resizing_converter(original_shape: tuple, new_shape: list) -> Callable:
    ox = original_shape[0]
    oy = original_shape[1]

    nx = new_shape[0]
    ny = new_shape[1]

    scale_x = ox / nx
    scale_y = oy / ny
    return lambda x, y: (scale_x * x, scale_y * y)


def bounding_box_to_actor_position(actor_polygon: Polygon) -> tuple:
    """bounding_box_to_actor_position.

    Converts the bounding box to the position in the image where the actor is on the ground plane
    (for now that is just the bottom of the bounding box)

    Args:
        actor_polygon (Polygon): actor_polygon

    Returns:
        tuple:
    """
    bottom_left, bottom_right = np.array(
        actor_polygon.exterior.coords[2]
    ), np.array(actor_polygon.exterior.coords[3])
    difference = bottom_right - bottom_left
    center = 0.5 * difference + bottom_left
    return center[0], center[1]
