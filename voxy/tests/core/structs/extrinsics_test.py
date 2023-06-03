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

from core.structs.extrinsics import Extrinsics


class ExtrinsicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.extrinsics = Extrinsics(5.0, 0.0, 0.0)

    def test_defaults(self) -> None:
        self.assertEqual(self.extrinsics.z_m, 5.0)

    def test_homogeneous_transform(self) -> None:
        transform = self.extrinsics.to_homogeneous_transform()
        a = 1
        b = 2
        c = 8
        point = np.array([a, b, c, 1]).reshape(4, 1)
        transformed_point = transform @ point
        print(transformed_point)
        self.assertEqual(transformed_point[0][0], a)
        self.assertEqual(transformed_point[1][0], self.extrinsics.z_m - c)
        self.assertEqual(transformed_point[2][0], b)

    def test_euler(self) -> None:
        rotated_extrinsics = Extrinsics(5.0, -np.pi / 2, 0.0)
        transform = rotated_extrinsics.to_homogeneous_transform()
        a = 1
        b = 2
        c = 8
        point = np.array([a, b, c, 1]).reshape(4, 1)
        point = transform @ point
        self.assertAlmostEqual(a, point[0][0], delta=1e-3)
        self.assertAlmostEqual(-b, point[1][0], delta=1e-3)
        self.assertAlmostEqual(
            self.extrinsics.z_m - c, point[2][0], delta=1e-3
        )


if __name__ == "__main__":
    unittest.main()
