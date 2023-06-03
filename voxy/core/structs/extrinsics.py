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

# trunk-ignore-all(mypy)
import attr
import numpy as np
from scipy.spatial.transform import Rotation


@attr.s(slots=True)
class Extrinsics:
    """Extrinsics model.

    The extrinsics are expressed in terms of the coordinate frame of the ground plane.
    The origin is defined directly below the camera with the x axis pointing to the right
    and the y axis pointing forward out in the direction of the camera's field of view
    """

    # this is the height of the camera off the ground expressed in meters
    z_m: float = attr.ib()

    # this is the roll of the camera expressed from the camera's frame
    # The units are in radians
    roll_rad: float = attr.ib()

    # this is the pitch of the camera from the camera's frame expressed in radians
    pitch_rad: float = attr.ib()

    def get_rotation_matrix(self) -> np.array:
        # this transforms the tranditional camera frame with x to the right,
        # y down and z out into the image
        # into conventional euclidean setup for the ground with z up and x, y lying an the plane
        # (y forward into the frame and x to the right)
        rotation_for_camera = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        rotation = Rotation.from_euler(
            "zyx", [0.0, self.pitch_rad, self.roll_rad]
        )
        return rotation_for_camera @ rotation.as_matrix()

    def to_homogeneous_transform(self) -> np.array:
        """Generate the homogeneous transform from world to camera.

        Args:

        Returns:
            np.array: the homogeneous transform (4x4 array)
        """
        world_T_camera = np.eye(4, dtype=float)
        world_T_camera[0:3, 0:3] = self.get_rotation_matrix()
        world_T_camera[2, 3] = self.z_m
        camera_T_world = np.linalg.inv(world_T_camera)
        return camera_T_world

    def get_camera_position(self) -> np.array:
        """get_camera_position.

        returns the camera position in the world frame
        (currently right below the camera)

        Args:

        Returns:
            np.array:
        """
        return np.array([0.0, 0.0, self.z_m])
