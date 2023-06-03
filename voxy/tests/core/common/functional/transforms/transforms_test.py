#
# Copyright 2023 Voxel Labs, Inc.
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

from core.common.functional.transforms.transforms import GetLabelYOLO
from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Polygon


class TransformsTest(unittest.TestCase):
    """
    Tests transforms in library
    """

    def test_actor_label_ordering_yolo(self) -> None:
        """Tests GetLabelYOLO to verify ordering of actor categories and
        verify if labels maintain order specified in the enum
        """
        image_height = 720
        image_width = 1280
        mocked_image = np.random.randn(image_height, image_width, 3)
        trailer = Actor(
            category=ActorCategory.TRAILER,
            polygon=Polygon.from_bbox([0, 0, 100, 100]),
        )
        person_v2 = Actor(
            category=ActorCategory.PERSON_V2,
            polygon=Polygon.from_bbox([50, 75, 130, 140]),
        )
        pit = Actor(
            category=ActorCategory.PIT,
            polygon=Polygon.from_bbox([650, 1000, 1280, 720]),
        )
        pit_v2 = Actor(
            category=ActorCategory.PIT_V2,
            polygon=Polygon.from_bbox([420, 200, 480, 250]),
        )
        actor_categories = ["TRAILER", "PERSON_V2", "PIT", "PIT_V2"]
        yolo_label_generator = GetLabelYOLO(actor_categories=actor_categories)
        self.assertTrue(
            yolo_label_generator.yolo_detector_classes
            == [
                ActorCategory.PIT,
                ActorCategory.PERSON_V2,
                ActorCategory.PIT_V2,
                ActorCategory.TRAILER,
            ]
        )
        generated_label = yolo_label_generator(mocked_image, trailer)
        gt_label = (
            f"3 {50.0 / image_width} {50.0 / image_height} "
            f"{100.0 / image_width} {100.0 / image_height}"
        )
        self.assertEqual(generated_label, gt_label)
        generated_label = yolo_label_generator(mocked_image, person_v2)
        gt_label = (
            f"1 {90.0 / image_width} {107.5 / image_height} "
            f"{80.0 / image_width} {65.0 / image_height}"
        )
        self.assertEqual(generated_label, gt_label)
        generated_label = yolo_label_generator(mocked_image, pit)
        gt_label = (
            f"0 {965.0 / image_width} {860.0 / image_height} "
            f"{630.0 / image_width} {280.0 / image_height}"
        )
        self.assertEqual(generated_label, gt_label)
        generated_label = yolo_label_generator(mocked_image, pit_v2)
        gt_label = (
            f"2 {450.0 / image_width} {225.0 / image_height} "
            f"{60.0 / image_width} {50.0 / image_height}"
        )
        self.assertEqual(generated_label, gt_label)
