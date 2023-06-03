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
from core.incident_machine.machines.overreaching import (
    ActorMachine,
    OverReachingViolationMachine,
)
from core.structs.actor import ActorCategory
from core.structs.state import PostureType, State


# trunk-ignore-all(pylint/E1101)
class OverReachingTest(unittest.TestCase):
    def test_transitions(self) -> None:
        """Test that the incident machine transitions work as expected"""
        # Test update_person_state trigger behavior
        actor_info = RelevantActorInfo(actor_id=101, track_uuid="101")
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 6000
        machine.update_person_state(person_reach_type=PostureType.BAD)
        self.assertEqual(machine.state, "reaching_badly")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")

        # Test update_person_state trigger behavior
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 300
        machine.update_person_state(person_reach_type=PostureType.BAD)
        self.assertEqual(machine.state, "reaching_badly")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 0)
        self.assertEqual(machine.state, "reaching_badly")
        machine.state_event_start_timestamp_ms = 1000
        machine.state_event_end_timestamp_ms = 7000
        machine.update_person_state(
            person_reach_type=PostureType.GOOD,
        )
        self.assertEqual(machine.state, "start")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 0)
        self.assertEqual(machine.state, "start")

    def test_state_and_events(self):
        """Tests handling of state and event messages with no actor supressions."""
        machine = OverReachingViolationMachine("", {})
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            person_reach_type=PostureType.GOOD,
            actor_category=ActorCategory.PERSON,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 0)

        _state = State(
            timestamp_ms=150_000,
            end_timestamp_ms=150_100,
            actor_id=actor_id,
            track_uuid=track_uuid,
            person_reach_type=PostureType.BAD,
            actor_category=ActorCategory.PERSON,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(
            machine.actor_machines[track_uuid].state, "reaching_badly"
        )

        _state = State(
            timestamp_ms=150_100,
            end_timestamp_ms=350_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            person_reach_type=PostureType.BAD,
            actor_category=ActorCategory.PERSON,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 1)

        _state = State(
            timestamp_ms=350_001,
            end_timestamp_ms=450_500,
            actor_id=actor_id,
            track_uuid=track_uuid,
            person_reach_type=PostureType.BAD,
            actor_category=ActorCategory.PERSON,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 1)
        incident = incidents[0]
        self.assertEqual(incident.cooldown_tag, True)


if __name__ == "__main__":
    unittest.main()
