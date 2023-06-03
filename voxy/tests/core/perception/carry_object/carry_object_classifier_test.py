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
from unittest.mock import patch

import numpy as np
import torch

from core.perception.carry_object.carry_object_classifier import (
    CarryObjectClassifier,
)
from core.structs.actor import Actor, ActorCategory, Polygon
from core.structs.attributes import KeyPoint

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class CarryObjectTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up Carry Object test...")

        # Create image
        self.image_shape = (720, 1280)
        rgb_range = 255
        n_color_channels = 3
        self.image = np.random.randint(
            rgb_range,
            size=(self.image_shape[0], self.image_shape[1], n_color_channels),
            dtype=np.uint8,
        )
        self.score_cutoff = 0.8
        self.prediction2class = {"NOT_CARRYING": 0, "CARRYING": 1}

        with patch(
            (
                "lib.ml.inference.tasks.ergonomic_carry_object.res_net34.factory."
                "Resnet34InferenceProviderFactory.get_inference_provider"
            )
        ) as mock_get_item:
            mock_get_item.return_value = "not_actually_loading_model"
            self.carry_object_classifier = CarryObjectClassifier(
                None,
                prediction2class=self.prediction2class,
                score_cutoff=self.score_cutoff,
                min_actor_pixel_area=1000,
                gpu_runtime=GpuRuntimeBackend.GPU_RUNTIME_BACKEND_LOCAL,
                triton_server_url="triton.server.name",
            )

    def test_filter_actor(self) -> None:
        """Tests filter actor logic"""
        # Test ActorCategory != PERSON

        actor = Actor(
            polygon=Polygon(
                [
                    KeyPoint(x=20, y=20),
                    KeyPoint(x=60, y=20),
                    KeyPoint(x=60, y=60),
                    KeyPoint(x=20, y=60),
                ]
            ),
            category=ActorCategory.PIT,
            confidence=0.5,
        )
        should_filter = self.carry_object_classifier.filter_actor(actor)
        self.assertTrue(should_filter)

        # Test confidence < 0.25
        actor = Actor(
            polygon=Polygon(
                [
                    KeyPoint(x=20, y=20),
                    KeyPoint(x=60, y=20),
                    KeyPoint(x=60, y=60),
                    KeyPoint(x=20, y=60),
                ]
            ),
            category=ActorCategory.PERSON,
            confidence=0.05,
        )
        should_filter = self.carry_object_classifier.filter_actor(actor)
        self.assertTrue(should_filter)

        # Test below pixel area thresh
        actor = Actor(
            polygon=Polygon(
                [
                    KeyPoint(x=20, y=20),
                    KeyPoint(x=30, y=20),
                    KeyPoint(x=30, y=30),
                    KeyPoint(x=20, y=30),
                ]
            ),
            category=ActorCategory.PERSON,
            confidence=0.5,
        )
        should_filter = self.carry_object_classifier.filter_actor(actor)
        self.assertTrue(should_filter)

        # Test valid
        actor = Actor(
            polygon=Polygon(
                [
                    KeyPoint(x=20, y=20),
                    KeyPoint(x=60, y=20),
                    KeyPoint(x=60, y=60),
                    KeyPoint(x=20, y=60),
                ]
            ),
            category=ActorCategory.PERSON,
            confidence=0.5,
        )
        should_filter = self.carry_object_classifier.filter_actor(actor)
        self.assertFalse(should_filter)

    def test_preprocess_inputs(self) -> None:
        """Tests overall pre-processing logic"""
        # 1 test to check the dimensions
        actor = Actor(
            polygon=Polygon(
                [
                    KeyPoint(x=20, y=20),
                    KeyPoint(x=60, y=20),
                    KeyPoint(x=60, y=60),
                    KeyPoint(x=20, y=60),
                ]
            )
        )
        cropped_images = self.carry_object_classifier.preprocess_inputs(
            [actor], self.image
        )

        self.assertEqual(cropped_images[0].shape, (3, 224, 224))

    def test_postprocess_model_outputs(self) -> None:
        """Test logic to overwrite positive predictions with negative
        (CARRYING TO NOT_CARRYING) lower than confidence threshold
        and that the results are returned in the same order as input.
        """
        # mismatched predictions and scores tensors, no change made to input predictions tensor
        predictions = torch.tensor(
            [
                self.prediction2class["CARRYING"],
                self.prediction2class["CARRYING"],
                self.prediction2class["NOT_CARRYING"],
                self.prediction2class["CARRYING"],
            ]
        )
        scores = torch.tensor(
            [self.score_cutoff - 0.1, self.score_cutoff + 0.1]
        )

        postprocessed_predictions = (
            self.carry_object_classifier.postprocess_model_outputs(
                scores, predictions
            )
        )
        self.assertEqual(
            torch.equal(postprocessed_predictions, predictions), True
        )

        # verify that low score predictions of positive class are overridden
        predictions = torch.tensor(
            [
                self.prediction2class["CARRYING"],
                self.prediction2class["NOT_CARRYING"],
                self.prediction2class["NOT_CARRYING"],
                self.prediction2class["CARRYING"],
            ]
        )
        scores = torch.tensor(
            [
                self.score_cutoff - 0.1,
                self.score_cutoff,
                self.score_cutoff + 0.1,
                self.score_cutoff + 0.2,
            ]
        )

        postprocessed_predictions = (
            self.carry_object_classifier.postprocess_model_outputs(
                scores, predictions
            )
        )
        self.assertEqual(
            postprocessed_predictions[0], self.prediction2class["NOT_CARRYING"]
        )
        # check that negative prediction is untouched
        self.assertEqual(
            postprocessed_predictions[1], self.prediction2class["NOT_CARRYING"]
        )


if __name__ == "__main__":
    unittest.main()
