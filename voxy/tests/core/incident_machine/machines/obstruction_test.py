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
from core.incident_machine.machines.obstruction import (
    ActorMachine,
    ObstructionMachine,
)
from core.structs.actor import ActorCategory
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ObstructionTest(unittest.TestCase):
    def test_transitions(self) -> None:
        """Test that the incident machine transitions work as expected"""
        # Test update_obstruction_state trigger behavior
        actor_info = RelevantActorInfo(actor_id=101, track_uuid="101")
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 130_000
        machine.update_obstruction_state(
            obstruction_is_stationary=True,
        )
        self.assertEqual(machine.state, "obstruction_in_area")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")

        # Test update_obstruction_state trigger behavior
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 1000
        machine.update_obstruction_state(
            obstruction_is_stationary=True,
        )
        self.assertEqual(machine.state, "obstruction_in_area")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 0)

        # Test check incident after max_incident_reset_time_ms trigger behavior
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 70000
        machine.update_obstruction_state(
            obstruction_is_stationary=True,
        )
        self.assertEqual(machine.state, "obstruction_in_area")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        machine.state_event_start_timestamp_ms = 70000
        machine.state_event_end_timestamp_ms = 80000
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        # After threshold
        machine.state_event_start_timestamp_ms = 700000
        machine.state_event_end_timestamp_ms = 720000
        machine.try_reset_state()
        machine.state_event_start_timestamp_ms = 700000
        machine.state_event_end_timestamp_ms = 800000
        machine.update_obstruction_state(
            obstruction_is_stationary=True,
        )

        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 2)

    def test_state_and_events(self):
        """Tests handling of state and event messages with no actor supressions."""
        machine = ObstructionMachine("", {})
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=120_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            obstruction_is_stationary=False,
            actor_category=ActorCategory.OBSTRUCTION,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")

        _state = State(
            timestamp_ms=120_000,
            end_timestamp_ms=122_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            obstruction_is_stationary=True,
            actor_category=ActorCategory.OBSTRUCTION,
            camera_uuid=None,
        )

        machine.process_state_event([_state])
        self.assertEqual(
            machine.actor_machines[track_uuid].state, "obstruction_in_area"
        )

        _state = State(
            timestamp_ms=120_000,
            end_timestamp_ms=240_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            obstruction_is_stationary=True,
            actor_category=ActorCategory.OBSTRUCTION,
            camera_uuid=None,
        )

        machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "incident")

        _state = State(
            timestamp_ms=841_000,
            end_timestamp_ms=860_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            obstruction_is_stationary=True,
            actor_category=ActorCategory.OBSTRUCTION,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])

        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 0)

        # Generate a cooldown incident when the actor based supression is turned off
        _state = State(
            timestamp_ms=841_000,
            end_timestamp_ms=1000_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            obstruction_is_stationary=True,
            actor_category=ActorCategory.OBSTRUCTION,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "incident")
        self.assertEqual(len(incidents), 1)
        incident = incidents[0]
        self.assertEqual(incident.cooldown_tag, True)


if __name__ == "__main__":
    unittest.main()
