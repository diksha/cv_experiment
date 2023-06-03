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

from core.perception.calibration.utils import (
    bounding_box_to_actor_position,
    calibration_config_to_camera_model,
)
from core.structs.frame import Frame


class WorldPerspectiveModel:
    def __init__(self, calibration_config: dict) -> None:
        # generate the camera model
        self.camera_model = calibration_config_to_camera_model(
            calibration_config
        )

    def __call__(self, frame: Frame) -> Frame:
        """__call__.

        process the current frame and apply any specific metadata

        (1): adds the current world position of the actors
        (2): adds the distance from the camera

        Args:
            frame (Frame): the current frame value

        Returns:
            Frame: the frame struct with calibration metadata information added
        """

        for actor_index, actor in enumerate(frame.actors):
            actor_bounding_box = actor.get_shapely_polygon()
            actor_u, actor_v = bounding_box_to_actor_position(
                actor_bounding_box
            )
            # TODO if we ever have height of the actor, we can use update that in the project image
            # point function here
            x_m, y_m = self.camera_model.project_image_point(actor_u, actor_v)
            z_m = 0.0

            # update the actor position
            frame.actors[actor_index].x_position_m = x_m
            frame.actors[actor_index].y_position_m = y_m
            frame.actors[actor_index].z_position_m = z_m

            camera_world = self.camera_model.extrinsics.get_camera_position()
            actor_position = np.array([x_m, y_m, z_m])
            distance_to_camera = np.linalg.norm(camera_world - actor_position)

            frame.actors[actor_index].distance_to_camera_m = distance_to_camera

        return frame
