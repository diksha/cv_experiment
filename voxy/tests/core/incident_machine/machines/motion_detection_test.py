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

from core.incident_machine.machines.base import RelevantActorInfo
from core.incident_machine.machines.motion_detection import (
    ActorMachine,
    MotionDetectionDowntimeMachine,
)
from core.structs.actor import ActorCategory
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class MotionDetectionTest(unittest.TestCase):
    def test_transitions(self) -> None:
        """Test that the incident machine transitions work as expected"""
        # Test update_motion_state trigger behavior
        actor_info = RelevantActorInfo(actor_id=101, track_uuid="101")
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "motion")
        machine.state_event_start_timestamp_ms = 0
        machine.state_event_end_timestamp_ms = 70_000
        machine.update_motion_state(is_in_motion=False)
        self.assertEqual(machine.state, "downtime")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")

        # Test update_motion_state trigger behavior
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "motion")
        machine.state_event_start_timestamp_ms = 0
        machine.state_event_end_timestamp_ms = 70_000
        machine.update_motion_state(is_in_motion=False)
        self.assertEqual(machine.state, "downtime")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")
        machine.state_event_start_timestamp_ms = 70_001
        machine.state_event_end_timestamp_ms = 80_000
        machine.update_motion_state(
            is_in_motion=True,
        )

        machine.try_reset_state()

        self.assertEqual(machine.state, "motion")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 0)
        self.assertEqual(machine.state, "motion")

    def test_state_and_events(self):
        """Tests handling of state and event messages with no actor supressions."""
        machine = MotionDetectionDowntimeMachine("", {})
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            motion_zone_is_in_motion=True,
            actor_category=ActorCategory.MOTION_DETECTION_ZONE,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "motion")
        self.assertEqual(len(incidents), 0)

        _state = State(
            timestamp_ms=150_000,
            end_timestamp_ms=300_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            motion_zone_is_in_motion=False,
            actor_category=ActorCategory.MOTION_DETECTION_ZONE,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "downtime")
        self.assertEqual(len(incidents), 1)
        incident1 = incidents[0]

        _state = State(
            timestamp_ms=300_000,
            end_timestamp_ms=450_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            motion_zone_is_in_motion=False,
            actor_category=ActorCategory.MOTION_DETECTION_ZONE,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "downtime")
        self.assertEqual(len(incidents), 1)
        incident2 = incidents[0]
        print(dir(incident2))
        self.assertEqual(incident2.cooldown_tag, True)

        # Test that incidents do not overlap in time.
        self.assertTrue(
            incident2.start_frame_relative_ms - incident1.end_frame_relative_ms
            >= 0
        )


if __name__ == "__main__":
    unittest.main()
