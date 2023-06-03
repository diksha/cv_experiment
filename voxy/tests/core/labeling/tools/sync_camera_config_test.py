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

import json
import unittest

import boto3
from moto import mock_s3

from core.labeling.tools.sync_camera_config import (
    get_normalized_points_for_scale,
    get_scale_label,
)
from core.utils.aws_utils import get_blobs_from_bucket


class SyncCameraConfigTest(unittest.TestCase):
    def test_get_normalized_points_for_scale(self) -> None:
        """Tests getting normalized points for scale annotations"""
        vertices = [{"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 1, "y": 2}]
        self.assertEqual(
            get_normalized_points_for_scale(vertices, 720.0, 1280.0, None),
            (
                [
                    [0.00078125, 0.001388888888888889],
                    [0.0015625, 0.002777777777777778],
                    [0.00078125, 0.002777777777777778],
                ],
                True,
            ),
        )

    def test_get_valid_polygon(self) -> None:
        """Tests getting a valid polygon (square)"""
        vertices = [
            {"x": 1, "y": 1},
            {"x": 1, "y": 2},
            {"x": 2, "y": 2},
            {"x": 2, "y": 1},
        ]
        self.assertEqual(
            get_normalized_points_for_scale(vertices, 720.0, 1280.0, None),
            (
                [
                    [0.00078125, 0.001388888888888889],
                    [0.00078125, 0.002777777777777778],
                    [0.0015625, 0.002777777777777778],
                    [0.0015625, 0.001388888888888889],
                ],
                True,
            ),
        )

    def test_get_invalid_polygon(self) -> None:
        """Tests getting a valid polygon (hourglass shape - self-intersecting)"""
        vertices = [
            {"x": 1, "y": 1},
            {"x": 2, "y": 2},
            {"x": 1, "y": 2},
            {"x": 2, "y": 1},
        ]
        self.assertEqual(
            get_normalized_points_for_scale(vertices, 720.0, 1280.0, None),
            (
                [
                    [0.00078125, 0.001388888888888889],
                    [0.0015625, 0.002777777777777778],
                    [0.00078125, 0.002777777777777778],
                    [0.0015625, 0.001388888888888889],
                ],
                False,
            ),
        )

    @mock_s3
    def test_get_scale_label(self) -> None:
        """Unit test for mocking the function get_scale_label and raising an error if the
        polygon is invalid.
        """
        s3_resource = boto3.resource("s3", region_name="us-east-1")
        s3_resource.create_bucket(Bucket="voxel-raw-labels-test")
        json_data = {
            "response": {
                "annotations": [
                    {
                        "label": "door",
                        "attributes": {
                            "type": "EXIT",
                            "orientation": "SIDE_DOOR",
                            "door_id": 1,
                        },
                        "uuid": "1134ab84-0cdf-4d7c-8ad6-109e5c8cddd6",
                        "vertices": [
                            {"x": 100, "y": 100},
                            {"x": 200, "y": 200},
                            {"x": 100, "y": 200},
                            {"x": 200, "y": 100},
                        ],
                        "type": "polygon",
                    }
                ],
                "camera_uuid": "ulta/dallas/0001/cha",
            },
            "image_shape": {"height": 720, "width": 1280},
        }
        s3object = s3_resource.Object(
            bucket_name="voxel-raw-labels-test",
            key="camera-config-test/org-test/site-test/camera-test/cha.json",
        )
        s3object.put(Body=(bytes(json.dumps(json_data).encode("UTF-8"))))
        blob = s3_resource.Bucket("voxel-raw-labels-test").objects.all()
        for blob in get_blobs_from_bucket(
            "voxel-raw-labels-test", "camera-config-test"
        ):
            with self.assertRaises(ValueError):
                get_scale_label(blob)

    @mock_s3
    def test_get_scale_label_actionable_regions(self) -> None:
        """Unit test for mocking the function get_scale_label and raising an error if the
        polygon is invalid.
        """
        s3_resource = boto3.resource("s3", region_name="us-east-1")
        s3_resource.create_bucket(Bucket="voxel-raw-labels-test")
        json_data = {
            "response": {
                "annotations": [
                    {
                        "label": "actionable_region",
                        "uuid": "1134ab84-0cdf-4d7c-8ad6-109e5c8cddd6",
                        "vertices": [
                            {"x": 100, "y": 100},
                            {"x": 200, "y": 200},
                            {"x": 100, "y": 200},
                            {"x": 200, "y": 100},
                        ],
                        "type": "polygon",
                    }
                ],
                "camera_uuid": "ulta/dallas/0001/cha",
            },
            "image_shape": {"height": 720, "width": 1280},
        }
        s3object = s3_resource.Object(
            bucket_name="voxel-raw-labels-test",
            key="camera-config-test/org-test/site-test/camera-test/cha.json",
        )
        s3object.put(Body=(bytes(json.dumps(json_data).encode("UTF-8"))))
        blob = s3_resource.Bucket("voxel-raw-labels-test").objects.all()
        for blob in get_blobs_from_bucket(
            "voxel-raw-labels-test", "camera-config-test"
        ):
            with self.assertRaises(ValueError):
                get_scale_label(blob)
