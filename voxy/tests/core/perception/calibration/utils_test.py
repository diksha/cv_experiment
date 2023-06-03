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

import yaml
from shapely.geometry import Polygon

from core.perception.calibration import utils
from core.perception.calibration.camera_model import CameraModel

example_yaml_calibration_config_string = """
calibration:
    intrinsics:
        # focal length in pixels based on a 299x299 image
        focal_length_pixels: 459.45546
        # distortion using the unified spherical model
        distortion_usm: 0.842461
        width_pixels: 100
        height_pixels: 100
        cx_pixels: 50
        cy_pixels: 50
    extrinsics:
        z_m: 6.199
        roll_radians: -0.3140
        pitch_radians: -0.0157
"""


class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_bounding_box_position(self) -> None:
        polygon = Polygon(
            [[482, 188], [482, 250], [572, 250], [572, 188], [482, 188]]
        )
        u, v = utils.bounding_box_to_actor_position(polygon)
        print(polygon)
        print((u, v))
        self.assertEqual(v, (250 + 188) / 2)
        self.assertEqual(u, 572)

    def test_yaml_load(self) -> None:
        config = yaml.safe_load(example_yaml_calibration_config_string)
        camera_model = utils.calibration_config_to_camera_model(config)
        self.assertTrue(isinstance(camera_model, CameraModel))

    def test_converter(self) -> None:
        converter = utils.resizing_converter((2000, 1000, 3), [500, 100])

        self.assertEqual(converter(250, 50), (1000, 500))


if __name__ == "__main__":
    unittest.main()
