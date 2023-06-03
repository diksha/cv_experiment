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

import torch

from lib.ml.inference.tasks.object_detection_2d.yolov5.pre_processing_model import (
    letterbox,
    preprocess_image,
)


class PreProcessingModelTest(unittest.TestCase):
    def test_preprocessing(self) -> None:
        """Test image preprocessing"""
        _input_shape = (480, 960)
        _new_shape = (480, 960)
        mock_image = torch.randint(
            low=0, high=255, size=(16, *_input_shape, 3), dtype=torch.uint8
        )
        new_shape = torch.tensor([[*_new_shape]]).expand([16, -1])
        self.assertTrue(letterbox is not None)
        self.assertTrue(preprocess_image is not None)
        preprocessed_batch, offset, scale = preprocess_image(
            mock_image,
            new_shape,
        )
        self.assertEqual(
            preprocessed_batch.shape, torch.Size([16, 3, *_new_shape])
        )
        self.assertTrue(
            torch.equal(offset, torch.tensor([[0.0, 0.0]]).expand([16, -1]))
        )
        self.assertTrue(
            torch.equal(scale, torch.tensor([[1.0, 1.0]]).expand([16, -1]))
        )
        _new_shape = (736, 1280)
        new_shape = torch.tensor([[*_new_shape]]).expand([16, -1])
        preprocessed_batch, offset, scale = preprocess_image(
            mock_image,
            new_shape,
        )
        self.assertEqual(
            preprocessed_batch.shape, torch.Size([16, 3, *_new_shape])
        )
        self.assertTrue(
            torch.equal(offset, torch.tensor([[0.0, 48.0]]).expand([16, -1]))
        )
        self.assertTrue(
            torch.equal(scale, torch.tensor([[4 / 3, 4 / 3]]).expand([16, -1]))
        )
