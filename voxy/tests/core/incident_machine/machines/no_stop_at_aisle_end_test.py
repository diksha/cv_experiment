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
from core.incident_machine.machines.no_stop_at_aisle_end import (
    ActorMachine,
    NoStopAtAisleMachine,
)
from core.structs.actor import ActorCategory
from core.structs.event import Event, EventType
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class NoStopAtAisleEndMachineTest(unittest.TestCase):
    def test_transitions(self) -> None:
        """Test that the incident machine transitions work as expected"""
        # Test update_pit_state trigger behavior
        actor_info = RelevantActorInfo(actor_id=101, track_uuid="101")
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.update_pit_state(is_pit_stationary=True)
        self.assertEqual(machine.state, "ever_seen_stopped")
        machine.update_pit_state(is_pit_stationary=True)
        self.assertEqual(machine.state, "ever_seen_stopped")
        machine.update_pit_state(is_pit_stationary=False)
        self.assertEqual(machine.state, "ever_seen_stopped")
        machine.try_reset_state()
        self.assertEqual(machine.state, "ever_seen_stopped")

        # Test pit_aisle_crossed trigger behavior
        incident_list = []
        machine = ActorMachine("", actor_info)
        machine.velocity_magnitude_at_aisle_crossing = (
            machine.INCIDENT_THRESH_LOW + 0.01
        )
        self.assertEqual(machine.state, "start")
        machine.pit_aisle_crossed(
            velocity=machine.velocity_magnitude_at_aisle_crossing,
            incident_list=incident_list,
        )
        self.assertEqual(machine.state, "incident")
        machine.try_reset_state()
        self.assertEqual(machine.state, "start")

        # Test pit_aisle_crossed trigger behavior
        incident_list = []
        machine = ActorMachine("", actor_info)
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 500
        self.assertEqual(machine.state, "start")
        machine.update_pit_state(is_pit_stationary=True)
        self.assertEqual(machine.state, "ever_seen_stopped")

        machine.state_event_start_timestamp_ms = 7000
        machine.state_event_end_timestamp_ms = 7000
        machine.velocity_magnitude_at_aisle_crossing = (
            machine.INCIDENT_THRESH_LOW + 0.01
        )
        machine.pit_aisle_crossed(
            velocity=machine.velocity_magnitude_at_aisle_crossing,
            incident_list=incident_list,
        )
        self.assertEqual(machine.state, "incident")
        machine.try_reset_state()
        self.assertEqual(machine.state, "start")

        # Test pit_aisle_crossed trigger behavior
        incident_list = []
        machine = ActorMachine("", actor_info)
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 500
        self.assertEqual(machine.state, "start")
        machine.update_pit_state(is_pit_stationary=True)
        self.assertEqual(machine.state, "ever_seen_stopped")

        machine.state_event_start_timestamp_ms = 1000
        machine.state_event_end_timestamp_ms = 1000
        machine.velocity_magnitude_at_aisle_crossing = (
            machine.INCIDENT_THRESH_LOW + 0.01
        )
        machine.pit_aisle_crossed(
            velocity=machine.velocity_magnitude_at_aisle_crossing,
            incident_list=incident_list,
        )
        self.assertEqual(machine.state, "start")
        machine.try_reset_state()
        self.assertEqual(machine.state, "start")

    def test_state_and_events(self):
        """Tests handling of state and event messages."""
        machine = NoStopAtAisleMachine("", {})
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=1000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            pit_is_stationary=False,
            actor_category=ActorCategory.PIT,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")

        _event = Event(
            timestamp_ms=101,
            subject_id=actor_id,
            object_id=None,
            end_timestamp_ms=101,
            normalized_speed=None,
            event_type=EventType.PIT_EXITING_AISLE,
            camera_uuid=None,
        )
        machine.process_state_event([_event])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")

        _event = Event(
            timestamp_ms=201,
            subject_id=actor_id,
            object_id=None,
            end_timestamp_ms=201,
            normalized_speed=1000,
            event_type=EventType.PIT_EXITING_AISLE,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_event])
        self.assertEqual(
            machine.actor_machines[track_uuid].state,
            "start",
        )
        self.assertTrue(len(incidents) > 0)

    def test_threshold_validation(self):
        """
        Tests that an incident is generated even if the INCIDENT_THRESH_HIGH
        is configured to be greater than INCIDENT_THRESH_LOW
        """
        machine = NoStopAtAisleMachine(
            "",
            {
                "incident_thresh_high": 0.5,
            },
        )
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=1000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            pit_is_stationary=False,
            actor_category=ActorCategory.PIT,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")

        _event = Event(
            timestamp_ms=101,
            subject_id=actor_id,
            object_id=None,
            end_timestamp_ms=101,
            normalized_speed=None,
            event_type=EventType.PIT_EXITING_AISLE,
            camera_uuid=None,
        )
        machine.process_state_event([_event])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")

        _event = Event(
            timestamp_ms=201,
            subject_id=actor_id,
            object_id=None,
            end_timestamp_ms=201,
            normalized_speed=0.51,
            event_type=EventType.PIT_EXITING_AISLE,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_event])
        self.assertEqual(
            machine.actor_machines[track_uuid].state,
            "start",
        )
        self.assertTrue(len(incidents) > 0)


if __name__ == "__main__":
    unittest.main()
