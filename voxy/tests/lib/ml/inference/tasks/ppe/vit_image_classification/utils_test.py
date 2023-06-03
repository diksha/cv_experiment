#
# Copyright 2020-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import typing
import unittest

import cv2
import numpy as np
import torch

from core.structs.actor import Actor
from core.structs.attributes import RectangleXYWH, RectangleXYXY
from lib.ml.inference.tasks.ppe.vit_image_classification.utils import (
    crop_actors,
)


class PreProcessingModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.actors = [
            Actor.from_dict(
                {
                    "category": "PERSON_V2",
                    "polygon": {
                        "vertices": [
                            {"x": 360.17, "y": 103.4, "z": None},
                            {"x": 370.4, "y": 103.4, "z": None},
                            {"x": 370.4, "y": 126.2, "z": None},
                            {"x": 360.17, "y": 126.2, "z": None},
                        ]
                    },
                }
            ),
            Actor.from_dict(
                {
                    "category": "PERSON_V2",
                    "polygon": {
                        "vertices": [
                            {"x": 480.17, "y": 200.4, "z": None},
                            {"x": 490.4, "y": 200.4, "z": None},
                            {"x": 490.4, "y": 226.2, "z": None},
                            {"x": 480.17, "y": 226.2, "z": None},
                        ]
                    },
                }
            ),
        ]
        self.image = np.random.uniform(
            low=0, high=255, size=(720, 1280, 3)
        ).astype("uint8")
        self.device = torch.device("cpu")

    def _legacy_crop_images(
        self,
        actors: typing.List[Actor],
        frame: np.ndarray,
        padding: int,
    ) -> torch.Tensor:
        cropped_images = []
        for actor in actors:
            rect = RectangleXYWH.from_polygon(actor.polygon)

            cropped_image = frame[
                max(0, rect.top_left_vertice.y - padding) : min(
                    frame.shape[0], rect.top_left_vertice.y + rect.h + padding
                ),
                max(0, rect.top_left_vertice.x - padding) : min(
                    frame.shape[1], rect.top_left_vertice.x + rect.w + padding
                ),
            ]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_images.append(cropped_image)

        return cropped_images

    def test_preprocess_inputs(self) -> None:
        actor_tensors = [
            torch.tensor(RectangleXYXY.from_polygon(actor.polygon).to_list())
            for actor in self.actors
        ]
        image_tensor = torch.from_numpy(self.image)
        for i in range(0, 500, 20):
            legacy_input = self._legacy_crop_images(
                self.actors,
                self.image,
                i,
            )
            updated_input = crop_actors(
                actor_tensors,
                image_tensor,
                i,
            )
            for i, legacy_crop in enumerate(legacy_input):
                updated_crop = updated_input[i]
                self.assertTrue(
                    torch.equal(torch.from_numpy(legacy_crop), updated_crop)
                )
