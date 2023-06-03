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
import unittest

import numpy as np

from core.perception.calibration.camera_model import CameraModel
from core.perception.calibration.models import unified_spherical_model
from core.structs.extrinsics import Extrinsics
from core.structs.intrinsics import Intrinsics


class CameraModelTest(unittest.TestCase):
    def setUp(self) -> None:
        focal_length = 299
        distortion = 0.0
        cx = 50.0
        cy = 50.0
        width = 100.0
        height = 100.0
        self.intrinsics = Intrinsics(
            focal_length, distortion, cx, cy, width, height
        )
        height_m = 0.0
        roll_radians = 0.0
        pitch_radians = 0.0
        self.extrinsics = Extrinsics(height_m, roll_radians, pitch_radians)
        self.camera_model = CameraModel(self.intrinsics, self.extrinsics)

    def test_camera_model(self) -> None:
        (ox, oy) = (25, 25)
        (ux, uy) = unified_spherical_model.undistort(ox, oy, self.intrinsics)
        # TODO: add a test to check for nonzero distortion
        self.assertEqual(ox, ux)
        self.assertEqual(oy, uy)

    def test_camera_projection(self) -> None:
        # project point
        extrinsics = Extrinsics(5.0, -np.pi / 4, 0.0)
        camera_model = CameraModel(self.intrinsics, extrinsics)
        original_x = 0.0  # m
        original_y = 5.0  # m
        # the geometry here is a pi/4 pi/4 pi/2 triangle
        # so this point should show up in the center of the image
        (u, v) = camera_model.project_world_point(original_x, original_y, 0)
        print(f"u: {u} v: {v} ")
        self.assertAlmostEqual(v, self.intrinsics.cy, delta=1e-3)
        self.assertTrue(isinstance(u, float))
        self.assertTrue(isinstance(v, float))
        (x, y) = camera_model.project_image_point(u, v)
        print(f"x: {x} y: {y} ")
        self.assertAlmostEqual(original_x, x, delta=1e-3)
        self.assertAlmostEqual(original_y, y, delta=1e-3)

    def test_world_velocity(self) -> None:
        # project point
        extrinsics = Extrinsics(5.0, -np.pi / 4, 0.0)
        camera_model = CameraModel(self.intrinsics, extrinsics)
        # the geometry here is a pi/4 pi/4 pi/2 triangle
        # so this point should show up in the center of the image
        (vx, vy) = camera_model.get_world_velocity(
            self.intrinsics.cx, self.intrinsics.cy, 5, 5
        )
        # gut check the velocities, it should be low since we are only at 5 px/s
        self.assertTrue(abs(vx) < 1)
        self.assertTrue(abs(vy) < 1)
        # gut check the velocities, it should be low since we are only at 5 px/s
        # vy is going to be moving toward the camera so it should be negative
        self.assertTrue(vy < 0)


if __name__ == "__main__":
    unittest.main()
