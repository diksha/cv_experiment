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

from core.incidents.controller import IncidentController
from core.structs.frame import Frame
from core.structs.incident import Incident


class IncidentControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = IncidentController(None)

    def test_generate_video_struct_10_fps(self):
        expected_frame_timestamps = [
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1600.0,
            1600.0,
            1600.0,
            1600.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            3000.0,
            3000.0,
            3000.0,
            3000.0,
            3000.0,
            3000.0,
            3000.0,
            3000.0,
            3000.0,
            3000.0,
        ]

        self.controller._INCIDENT_VIDEO_FPS = 10
        self.controller._frame_struct_buffer = {
            0.0: Frame.from_dict({"relative_timestamp_ms": 0.0}),
            1000.0: Frame.from_dict({"relative_timestamp_ms": 1000.0}),
            1600.0: Frame.from_dict({"relative_timestamp_ms": 1600.0}),
            2000.0: Frame.from_dict({"relative_timestamp_ms": 2000.0}),
            3000.0: Frame.from_dict({"relative_timestamp_ms": 3000.0}),
            4000.0: Frame.from_dict({"relative_timestamp_ms": 4000.0}),
            5000.0: Frame.from_dict({"relative_timestamp_ms": 5000.0}),
            6000.0: Frame.from_dict({"relative_timestamp_ms": 6000.0}),
            7000.0: Frame.from_dict({"relative_timestamp_ms": 7000.0}),
        }
        incident = Incident(
            start_frame_relative_ms=1000.0, end_frame_relative_ms=4000.0
        )
        video = self.controller._generate_video_struct(incident)
        frame_timestamps = []
        for frame in video.frames:
            frame_timestamps.append(frame.relative_timestamp_ms)
        self.assertEqual(expected_frame_timestamps, frame_timestamps)


if __name__ == "__main__":
    unittest.main()
