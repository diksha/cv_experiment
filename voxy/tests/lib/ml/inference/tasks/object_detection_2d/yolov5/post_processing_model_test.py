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

from lib.ml.inference.tasks.object_detection_2d.yolov5.post_processing_model import (
    pad_observations,
    transform_and_post_process,
    unpack_observations,
)


class PostProcessingModelTest(unittest.TestCase):
    def test_postprocessing_compiles(self) -> None:
        """Test image postprocessing"""
        mock_prediction = torch.tensor(
            [
                [
                    [4.8320, 6.8086, 8.8203, 12.4375, 0.0000, 0.6250, 0.3386],
                    [
                        15.2734,
                        7.2773,
                        21.7969,
                        13.4453,
                        0.0000,
                        0.4841,
                        0.5054,
                    ],
                    [
                        21.7812,
                        7.6328,
                        27.9375,
                        11.9219,
                        0.0000,
                        0.7104,
                        0.3005,
                    ],
                    [30.6250, 6.2930, 18.8281, 9.6016, 0.0000, 0.8735, 0.1240],
                ]
            ],
            device="cpu",
            dtype=torch.float16,
        )
        for _ in range(4):
            mock_prediction = torch.cat(
                [mock_prediction, mock_prediction], dim=1
            )
        offset, scale = (158.0, 0.0), (0.6486486486486487, 0.6486486486486487)
        self.assertTrue(transform_and_post_process is not None)
        # TODO load model and run end to end test
        self.assertTrue(unpack_observations is not None)
        self.assertTrue(
            transform_and_post_process(
                mock_prediction,
                torch.tensor([offset]),
                torch.tensor([scale]),
                torch.tensor([[0, 1]]),
                torch.tensor([[0.001]]),
                torch.tensor([[0.7]]),
            )
            is not None
        )

    def test_pad_observations(self):
        """
        Tests padding of observations
        """
        observation_list = [
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ]
        expected = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ]
        )
        actual = pad_observations(observation_list)
        self.assertTrue(torch.all(torch.eq(expected, actual)))

        # test empty tensor
        observation_list = [
            torch.empty((0, 3)),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ]
        expected = torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ]
        )
        actual = pad_observations(observation_list)
        self.assertTrue(torch.all(torch.eq(expected, actual)))
        # test all empty
        observation_list = [
            torch.empty((0, 3)),
            torch.empty((0, 3)),
            torch.empty((0, 3)),
        ]
        expected = torch.tensor(
            [
                [[0, 0, 0]],
                [[0, 0, 0]],
                [[0, 0, 0]],
            ]
        )
        actual = pad_observations(observation_list)
        self.assertTrue(torch.all(torch.eq(expected, actual)))

    def test_unpack(self):
        observation = torch.tensor(
            [
                [
                    [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                    [2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                    [2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                    [2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                    [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                ],
                [
                    [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                    [2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                    [2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                    [2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                    [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
                ],
            ]
        )
        num_observations = torch.tensor([5, 5])
        classes = {1: "class1", 2: "class2"}
        results = unpack_observations(observation, num_observations, classes)
        self.assertTrue(len(results) == 2)
        self.assertTrue(len(results[0]) == 2)
        self.assertTrue(len(results[1]) == 2)
        self.assertTrue(results[0]["class1"].shape == (2, 7))
        self.assertTrue(results[0]["class2"].shape == (3, 7))
        self.assertTrue(results[1]["class1"].shape == (2, 7))
        self.assertTrue(results[1]["class2"].shape == (3, 7))
