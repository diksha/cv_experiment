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

import cv2
import torch

from core.structs.actor import ActorCategory
from lib.infra.utils.resolve_model_path import resolve_model_path
from lib.ml.inference.backends.trt import TRTBackend
from lib.ml.inference.tasks.object_detection_2d.yolov5.post_processing_model import (
    transform_and_post_process,
    unpack_observations,
)
from lib.ml.inference.tasks.object_detection_2d.yolov5.utils import (
    preprocess_image,
)


class EndToEndPostProcessingModelTest(unittest.TestCase):
    def test_postprocessing_end_to_end(self) -> None:

        # Get the list of all files and directories
        # in current working directory
        model_path = resolve_model_path(
            "artifacts_2021-12-06-00-00-00-0000-yolo/best_480_960.engine"
        )

        image_path = resolve_model_path(
            "artifacts_test_image_inference/test_image.png"
        )
        test_image = cv2.imread(image_path)
        sample_input_shape = [480, 960]  # height, width from configs

        batched_input = torch.from_numpy(test_image).unsqueeze(0)
        processed, offset, scale = preprocess_image(
            batched_input, sample_input_shape, "cuda"
        )
        backend = TRTBackend(engine_file_path=model_path, device="cuda")
        prediction = backend.infer([processed])[0]
        confidence_threshold = 0.001
        nms_threshold = 0.7
        post_processed = transform_and_post_process(
            prediction,
            torch.tensor([offset]),
            torch.tensor([scale]),
            torch.tensor([[0, 1]]),
            torch.tensor([[confidence_threshold]]),
            torch.tensor([[nms_threshold]]),
        )
        self.assertTrue(isinstance(post_processed, tuple))
        print(post_processed)
        print(post_processed[0])
        print(post_processed[1].size())
        unpack_observed = unpack_observations(
            post_processed[1],
            post_processed[0],
            {0: ActorCategory.PERSON, 1: ActorCategory.PIT},
        )
        self.assertTrue(len(unpack_observed) == 1)
        self.assertTrue(isinstance(unpack_observed[0], dict))
