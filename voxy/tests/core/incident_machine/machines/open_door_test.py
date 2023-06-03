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
from core.incident_machine.machines.open_door import (
    ActorMachine,
    OpenDoorMachine,
)
from core.structs.actor import ActorCategory, DoorState
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class OpenDoorTest(unittest.TestCase):
    def test_transitions(self) -> None:
        """Test that the incident machine transitions work as expected"""
        # Test update_door_state trigger behavior
        actor_info = RelevantActorInfo(actor_id=101, track_uuid="101")
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "closed")
        machine.state_event_start_timestamp_ms = 0
        machine.state_event_end_timestamp_ms = 70_000
        machine.update_door_state(
            door_state=DoorState.FULLY_OPEN,
        )
        self.assertEqual(machine.state, "open")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")

        # Test update_door_state trigger behavior
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "closed")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 1000
        machine.update_door_state(
            door_state=DoorState.FULLY_OPEN,
        )
        self.assertEqual(machine.state, "open")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 0)
        self.assertEqual(machine.state, "open")
        machine.state_event_start_timestamp_ms = 1000
        machine.state_event_end_timestamp_ms = 7000
        machine.update_door_state(
            door_state=DoorState.FULLY_CLOSED,
        )
        self.assertEqual(machine.state, "closed")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 0)
        self.assertEqual(machine.state, "closed")

    # Adding tests for 20 minute open door incidents
    def test_long_transitions(self) -> None:
        """
        Test that the incident machine state transitions work as expected for long (20 minutes) open
        door duration having multiple state_event_messages and checking if it generates an incident.
        """
        # Test incident machine behavior for partially_open > 20 minutes
        actor_info = RelevantActorInfo(actor_id=0, track_uuid="00")
        machine = ActorMachine("", actor_info, {"max_open_door_s": 1200})

        # state_event_message: DoorState.PARTIALLY_OPEN
        state_event_message = State(
            timestamp_ms=0,
            camera_uuid="org/site/000x/cha",
            actor_id="0",
            actor_category=ActorCategory.DOOR,
            end_timestamp_ms=120_000,
            door_state=DoorState.PARTIALLY_OPEN,
        )
        incident_list = machine.transition_state_event(state_event_message)
        self.assertEqual(machine.state, "partially_open")
        self.assertEqual(incident_list, None)

        # state_event_message:DoorState.FULLY_OPEN
        state_event_message = State(
            timestamp_ms=119_900,
            camera_uuid="org/site/000x/cha",
            actor_id="0",
            actor_category=ActorCategory.DOOR,
            end_timestamp_ms=1000_000,
            door_state=DoorState.FULLY_OPEN,
        )
        incident_list = machine.transition_state_event(state_event_message)
        self.assertEqual(machine.state, "open")
        self.assertEqual(incident_list, None)

        # state_event_message: DoorState.FULLY_OPEN & Generate Incident
        state_event_message = State(
            timestamp_ms=999_000,
            camera_uuid="org/site/000x/cha",
            actor_id="0",
            actor_category=ActorCategory.DOOR,
            end_timestamp_ms=1321_000,
            door_state=DoorState.FULLY_OPEN,
        )
        incident_list = machine.transition_state_event(state_event_message)
        self.assertEqual(
            machine.state, "open"
        )  # machine.try_reset_state will make it open again!
        self.assertNotEqual(
            incident_list, None
        )  # It uses "assertNotEqual" to imply incident is not None!

        # state_event_message: DoorState.FULLY_CLOSED
        state_event_message = State(
            timestamp_ms=1320_900,
            camera_uuid="org/site/000x/cha",
            actor_id="0",
            actor_category=ActorCategory.DOOR,
            end_timestamp_ms=1322_000,
            door_state=DoorState.FULLY_CLOSED,
        )
        incident_list = machine.transition_state_event(state_event_message)
        self.assertEqual(machine.state, "closed")
        self.assertEqual(incident_list, None)

    def test_state_and_events(self):
        """Tests handling of state and event messages with no actor supressions."""
        machine = OpenDoorMachine("", {})
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            door_state=DoorState.FULLY_CLOSED,
            actor_category=ActorCategory.DOOR,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "closed")

        _state = State(
            timestamp_ms=150_000,
            end_timestamp_ms=200_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            door_state=DoorState.FULLY_OPEN,
            actor_category=ActorCategory.DOOR,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "open")

        _state = State(
            timestamp_ms=200_000,
            end_timestamp_ms=350_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            door_state=DoorState.FULLY_OPEN,
            actor_category=ActorCategory.DOOR,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "open")
        self.assertEqual(len(incidents), 1)

        _state = State(
            timestamp_ms=350_001,
            end_timestamp_ms=600_500,
            actor_id=actor_id,
            track_uuid=track_uuid,
            door_state=DoorState.FULLY_OPEN,
            actor_category=ActorCategory.DOOR,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "open")
        self.assertEqual(len(incidents), 1)
        incident = incidents[0]
        self.assertEqual(incident.cooldown_tag, True)


if __name__ == "__main__":
    unittest.main()
