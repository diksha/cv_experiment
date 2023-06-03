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

import mock
import numpy as np

from core.labeling.tools.utils import validate_camera_config_response


class UtilsTest(unittest.TestCase):
    @mock.patch("core.labeling.tools.utils.upload_directory_to_s3")
    @mock.patch("core.labeling.tools.utils.get_frame_from_kinesis")
    def test_validate_camera_config_response(
        self, mock_get_frame_from_kinesis, mock_upload_directory_to_s3
    ) -> None:
        mock_get_frame_from_kinesis.return_value = np.random.rand(3, 2)
        polygon = [{"polygon": [[1, 2], [2, 3], [3, 4], [4, 5]]}]
        camera_config_response = {
            "camera_uuid": {
                "cameraConfigNew": {
                    "doors": json.dumps(polygon),
                    "drivingAreas": "[]",
                    "actionableRegions": "[]",
                    "intersections": "[]",
                    "endOfAisles": "[]",
                    "noPedestrianZones": "[]",
                    "motionDetectionZones": "[]",
                    "noObstructionRegions": "[]",
                },
                "isUpdated": True,
            }
        }
        validate_camera_config_response(camera_config_response)
        mock_upload_directory_to_s3.assert_called_once()
