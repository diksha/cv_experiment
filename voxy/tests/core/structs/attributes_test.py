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

from core.structs.attributes import KeyPoint, Pose

# trunk-ignore-begin(pylint/E0611)
from protos.perception.types.v1.types_pb2 import KeyPoint as KeyPointPb

# trunk-ignore-end(pylint/E0611)


class PoseTest(unittest.TestCase):
    """Unit Test for Pose Serialization"""

    def setUp(self) -> None:
        """Set up test"""
        keypoint1 = KeyPointPb(
            x=2.0,
            y=3.0,
            confidence_probability=0.95,
        )
        keypoint2 = KeyPointPb(
            x=2.0,
            y=5.0,
            confidence_probability=0.95,
        )
        keypoint3 = KeyPointPb(
            x=4.0,
            y=5.0,
            confidence_probability=0.95,
        )
        self.pose = Pose(
            nose=KeyPoint.from_proto(keypoint1),
            neck=KeyPoint.from_proto(keypoint2),
            right_shoulder=KeyPoint.from_proto(keypoint3),
        )

    def test_serialization(self):
        """Test serialization"""
        json_serialized = json.dumps(self.pose.to_dict())
        pose_from_json = Pose.from_dict(json.loads(json_serialized))
        self.assertEqual(self.pose, pose_from_json)
        proto_serialized = self.pose.to_proto()
        pose_from_proto = Pose.from_proto(proto_serialized)
        self.assertEqual(self.pose, pose_from_proto)
