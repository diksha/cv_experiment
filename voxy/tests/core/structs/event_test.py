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
import uuid

from core.structs.event import Event, EventType

# trunk-ignore(pylint/E0611)
from core.structs.protobufs.v1.event_pb2 import Event as EventPb

# trunk-ignore-all(pylint/C0116)


class EventTest(unittest.TestCase):
    def setUp(self) -> None:
        self.event = Event(
            1, "", "", EventType.PIT_ENTERING_DOOR, "", 5, "", 5.0, 2.0, 0.0
        )

    def test_to_proto(self) -> None:
        self.assertTrue(self.event.to_proto() is not None)
        self.assertTrue(isinstance(self.event.to_proto(), EventPb))

    def test_grouping_key(self) -> None:
        self.assertTrue(self.event.grouping_key is not None)

    def test_differentiator(self) -> None:
        self.assertTrue(self.event.differentiator is not None)
        self.assertTrue(isinstance(self.event.differentiator, tuple))

    def test_proto_roundtrip(self) -> None:
        events = [
            # test with all fields populated
            Event(
                timestamp_ms=10,
                camera_uuid="fake/camera/0001/cha",
                subject_id=str(uuid.uuid4()),
                event_type=EventType.DOOR_CLOSED,
                object_id=str(uuid.uuid4()),
                end_timestamp_ms=20,
                run_uuid=str(uuid.uuid4()),
                x_velocity_pixel_per_sec=1.0,
                y_velocity_pixel_per_sec=2.0,
                normalized_speed=5.0,
                subject_uuid=str(uuid.uuid4()),
                object_uuid=str(uuid.uuid4()),
            ),
            # test with some none fields
            Event(
                timestamp_ms=10,
                camera_uuid="fake/camera/0001/cha",
                subject_id=str(uuid.uuid4()),
                event_type=EventType.DOOR_CLOSED,
                object_id=str(uuid.uuid4()),
                end_timestamp_ms=20,
                run_uuid=str(uuid.uuid4()),
                x_velocity_pixel_per_sec=None,
                y_velocity_pixel_per_sec=None,
                normalized_speed=None,
                subject_uuid=str(uuid.uuid4()),
                object_uuid=str(uuid.uuid4()),
            ),
        ]

        for event in events:
            self.assertEqual(event, Event.from_proto(event.to_proto()))


if __name__ == "__main__":
    unittest.main()
