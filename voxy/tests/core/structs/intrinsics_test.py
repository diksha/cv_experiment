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

from core.structs.intrinsics import Intrinsics


class IntrinsicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = Intrinsics(299, 0.0, 50, 50, 100, 100)

    def test_defaults(self) -> None:
        self.assertEqual(self.intrinsics.cx, 50)
        self.assertEqual(self.intrinsics.cy, 50)
        # TODO add test for rest of intrinsics functionality

    def test_focal_length(self) -> None:
        matrix = self.intrinsics.to_calibration_matrix()
        fx = matrix[0, 0]
        self.assertAlmostEqual(
            np.arctan2(fx, self.intrinsics.width_px / 2), np.pi / 4, delta=1e-5
        )
        fy = matrix[1, 1]
        self.assertAlmostEqual(
            np.arctan2(fy, self.intrinsics.height_px / 2),
            np.pi / 4,
            delta=1e-5,
        )

    # TODO: add intrinsics matrix test


if __name__ == "__main__":
    unittest.main()
