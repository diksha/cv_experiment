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

from core.structs.attributes import Point, Polygon
from core.utils.struct_utils.utils import get_bbox_from_xysr, get_iou


class StructUtilsTest(unittest.TestCase):
    def test_iou(self) -> None:
        polygon_1 = Polygon(
            vertices=[Point(1, 1), Point(1, 2), Point(2, 2), Point(2, 1)]
        )
        polygon_2 = Polygon(
            vertices=[
                Point(1, 1),
                Point(1, 1.5),
                Point(1.5, 1.5),
                Point(1.5, 1),
            ]
        )

        self.assertEqual(get_iou(polygon_1, polygon_1), 1)
        self.assertEqual(get_iou(polygon_1, polygon_2), 0.25)

    def test_get_bbox_from_xysr(self) -> None:
        xysr = np.array(
            [
                [10, 20, 100, 1],
                [0, 5, 100, 1],
            ]
        )
        bbox_expected = np.array(
            [
                [5, 15, 15, 25],
                [-5, 0, 5, 10],
            ]
        )
        bbox = get_bbox_from_xysr(xysr)
        self.assertTrue(np.allclose(bbox, bbox_expected))
        xysr = np.array(
            [10, 20, 100, 1],
        )
        bbox_expected = np.array(
            [5, 15, 15, 25],
        )
        bbox = get_bbox_from_xysr(xysr)
        self.assertTrue(np.allclose(bbox, bbox_expected))
