#
# Copyright 2020-2022 Voxel Labs, Inc.
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

from core.perception.common.utils import reshape_polygon_crop_to_square
from core.structs.actor import Polygon
from core.structs.attributes import KeyPoint


class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up perception utils test...")

        # Create image
        self.image_shape = (720, 1280)
        rgb_range = 255
        n_color_channels = 3
        self.image = np.random.randint(
            rgb_range,
            size=(self.image_shape[0], self.image_shape[1], n_color_channels),
            dtype=np.uint8,
        )

    def test_reshape_polygon_crop_to_square(self) -> None:
        """Tests reshape_polygon_crop_to_square utility function"""
        # Test height > width
        polygon = Polygon(
            [
                KeyPoint(x=20, y=20),
                KeyPoint(x=30, y=20),
                KeyPoint(x=30, y=40),
                KeyPoint(x=20, y=40),
            ]
        )
        (
            from_0,
            to_0,
            from_1,
            to_1,
        ) = reshape_polygon_crop_to_square(polygon, self.image)

        self.assertEqual(to_0 - from_0, to_1 - from_1)
        self.assertEqual(to_0 - from_0, 40)
        self.assertEqual(
            [
                from_0,
                to_0,
                from_1,
                to_1,
            ],
            [
                10,
                50,
                5,
                45,
            ],
        )

        # Test width > height
        polygon = Polygon(
            [
                KeyPoint(x=20, y=60),
                KeyPoint(x=50, y=60),
                KeyPoint(x=50, y=70),
                KeyPoint(x=20, y=70),
            ]
        )
        (
            from_0,
            to_0,
            from_1,
            to_1,
        ) = reshape_polygon_crop_to_square(polygon, self.image)

        self.assertEqual(to_0 - from_0, to_1 - from_1)
        self.assertEqual(to_0 - from_0, 50)
        self.assertEqual(
            [
                from_0,
                to_0,
                from_1,
                to_1,
            ],
            [
                40,
                90,
                10,
                60,
            ],
        )

        # Test width = height
        polygon = Polygon(
            [
                KeyPoint(x=20, y=20),
                KeyPoint(x=60, y=20),
                KeyPoint(x=60, y=60),
                KeyPoint(x=20, y=60),
            ]
        )
        (
            from_0,
            to_0,
            from_1,
            to_1,
        ) = reshape_polygon_crop_to_square(polygon, self.image)

        self.assertEqual(to_0 - from_0, to_1 - from_1)
        self.assertEqual(to_0 - from_0, 60)
        self.assertEqual(
            [
                from_0,
                to_0,
                from_1,
                to_1,
            ],
            [
                10,
                70,
                10,
                70,
            ],
        )

        # Test out of bounds top left
        polygon = Polygon(
            [
                KeyPoint(x=0, y=0),
                KeyPoint(x=40, y=0),
                KeyPoint(x=40, y=40),
                KeyPoint(x=0, y=40),
            ]
        )
        (
            from_0,
            to_0,
            from_1,
            to_1,
        ) = reshape_polygon_crop_to_square(polygon, self.image)

        self.assertEqual(to_0 - from_0, to_1 - from_1)
        self.assertEqual(to_0 - from_0, 60)
        self.assertEqual(
            [
                from_0,
                to_0,
                from_1,
                to_1,
            ],
            [
                0,
                60,
                0,
                60,
            ],
        )

        # Test out of bounds bottom right
        polygon = Polygon(
            [
                KeyPoint(x=1240, y=680),
                KeyPoint(x=1280, y=680),
                KeyPoint(x=1280, y=720),
                KeyPoint(x=1240, y=720),
            ]
        )
        (
            from_0,
            to_0,
            from_1,
            to_1,
        ) = reshape_polygon_crop_to_square(polygon, self.image)

        self.assertEqual(to_0 - from_0, to_1 - from_1)
        self.assertEqual(to_0 - from_0, 60)
        self.assertEqual(
            [
                from_0,
                to_0,
                from_1,
                to_1,
            ],
            [
                660,
                720,
                1220,
                1280,
            ],
        )


if __name__ == "__main__":
    unittest.main()
