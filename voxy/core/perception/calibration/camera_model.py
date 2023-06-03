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

from core.perception.calibration.models import unified_spherical_model
from core.structs.extrinsics import Extrinsics
from core.structs.intrinsics import Intrinsics


class CameraModel:
    """CameraModel.

    Model of the camera based on the intrinsic distortion model and world coordinate frame
    """

    def __init__(self, intrinsics: Intrinsics, extrinsics: Extrinsics):
        """__init__.

        Args:
            intrinsics (Intrinsics): intrinsics
            extrinsics (Extrinsics): extrinsics
        """
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        # generate calibration matrix
        self.calibration_matrix = self.intrinsics.to_calibration_matrix()
        # generate camera matrix
        self.camera_T_world = self.extrinsics.to_homogeneous_transform()
        self.camera_rotation = self.extrinsics.get_rotation_matrix()
        self.camera_matrix = self.calibration_matrix @ self.camera_T_world

    def project_world_point(self, x_m: float, y_m: float, z_m: float) -> tuple:
        """Projects world points into the image frame

        Args:
            x_m: the x position in meters
            y_m: the y position in meters
            z_m: the z position in meters

        Returns:
            tuple: the projection of the world point in image coordindates (u, v)
        """
        # get the point in inhomogeneous coordinates
        world_point = np.array([x_m, y_m, z_m, 1.0]).reshape(4, 1)
        image_point = self.camera_matrix @ world_point
        # apply perspective to get the point back in inhomogeneous coordinates
        image_point = np.multiply(image_point, 1 / image_point[2][0])
        return (image_point[0][0], image_point[1][0])

    def project_image_point(self, u: float, v: float) -> tuple:
        """project_image_point.

        projects image point on to the ground plane given by the intrinsics and extrinsics

            1. Apply the inverse distortion model to undistort the image points. This puts them into a pinhole camera geometry.
            2. Transform the image coordinate to a homogeneous coordinate with weight 𝑤. Any real, non-zero 𝑤 may be used, however two common ones are 1 or the distance from the camera center to the world point.
            3. Multiply the homogeneous image coordinate by the inverse of the intrinsic camera matrix. This vector is the direction vector of the line between the point and the camera center in camera-relative coordinates.
            4. Apply the rotation of the camera’s pose (inverse of the point transform matrix) to the direction vector. Separately, compute the camera center to form a point on the 3D line.

        Args:
            u (float): u: image coordinate row
            v (float): v: image coordinate column

        Returns:
            tuple: the x, y position in meters on the plane
        """
        #   reference: https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/
        #   1. Apply the inverse distortion model to undistort the image points. This puts them into a pinhole camera geometry.

        # TODO(twroge): undistort using intrinsics
        (u, v) = unified_spherical_model.undistort(u, v, self.intrinsics)

        # project points
        #   2. Transform the image coordinate to a homogeneous coordinate with weight 𝑤.
        #      Any real, non-zero 𝑤 may be used, however two common ones are 1 or the distance
        #      from the camera center to the world point.
        image_point = np.array([u, v, 1, 1]).reshape(4, 1)

        #   3. Multiply the homogeneous image coordinate by the inverse of the intrinsic camera
        #      matrix. This vector is the direction vector of the line between the point and
        #      the camera center in camera-relative coordinates.
        camera_image_line = (
            np.linalg.inv(self.calibration_matrix) @ image_point
        )

        # the camera image line defines a point in the camera frame
        # we can convert the image point to world coordinates
        image_point_world = (
            np.linalg.inv(self.extrinsics.to_homogeneous_transform())
            @ camera_image_line
        )

        # the line is defined by
        # camera_origin_world + t * (camera_origin_world - image_point_world)
        # that gives us one degree of freedom and we can solve using
        # the fact that the world point will be on the ground
        camera_origin_world = np.array(
            [0.0, 0, self.extrinsics.z_m, 1]
        ).reshape(4, 1)
        z0 = camera_origin_world[2][0]
        z1 = image_point_world[2][0]
        t = z0 / (z1 - z0)
        world_point = camera_origin_world + np.multiply(
            t, (camera_origin_world - image_point_world)
        )
        # TODO(twroge): add check for points at infinity
        return (world_point[0][0], world_point[1][0])

    def get_world_velocity(
        self,
        u: float,
        v: float,
        x_velocity_pixels_per_second: float,
        y_velocity_pixels_per_second: float,
    ) -> tuple:
        """get_world_velocity.

        This generates world velocity for the points in the image. The assumption is that the actor
        is moving completely on the ground plane

        Args:
            u: the u coordinate of the actor in the image (full resolution)
            v: the v coordinate of the actor in the image (full resolution)
            x_velocity_pixels_per_second: the pixel velocity of the agent (x component) (full resolution)
            y_velocity_pixels_per_second: the pixel velocity of the agent (y component) (full resolution)

        Returns:
            tuple: world velocity in the x direction and y direction according to the axis convention in
                   /core/structs/extrinsics.py
        """

        # TODO: compute this world velocity using the jacobian of p(x)  -> v(x)
        current_x_m, current_y_m = self.project_image_point(u, v)
        # see where the object will be at the next time step
        future_x_m, future_y_m = self.project_image_point(
            u + x_velocity_pixels_per_second, v + y_velocity_pixels_per_second
        )

        finite_mask_current = np.isfinite([current_x_m, current_y_m])
        finite_mask_future = np.isfinite([future_x_m, future_y_m])

        if not np.all(finite_mask_current):
            print(finite_mask_current)
            print(
                "[get_world_velocity] Warning! current actor location was beyond horizon"
            )
            print("[get_world_velocity] Returning pixel velocity")
            return (x_velocity_pixels_per_second, y_velocity_pixels_per_second)

        if not np.all(finite_mask_future):
            print(finite_mask_future)
            print(
                "[get_world_velocity] Warning! future actor location was beyond horizon"
            )
            print("[get_world_velocity] Returning pixel velocity")
            return (x_velocity_pixels_per_second, y_velocity_pixels_per_second)

        dx_dt = future_x_m - current_x_m
        dy_dt = future_y_m - current_y_m

        return (dx_dt, dy_dt)
