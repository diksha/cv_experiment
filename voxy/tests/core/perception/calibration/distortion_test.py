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

from core.perception.calibration.models import unified_spherical_model
from core.structs.intrinsics import Intrinsics


class DistortionModelTest(unittest.TestCase):
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

    def test_distortion(self) -> None:
        (ox, oy) = (25, 25)
        (ux, uy) = unified_spherical_model.undistort(ox, oy, self.intrinsics)
        # TODO: add a test to check for nonzero distortion
        self.assertEqual(ox, ux)
        self.assertEqual(oy, uy)


if __name__ == "__main__":
    unittest.main()
