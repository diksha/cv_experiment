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
from core.incident_machine.machines.n_person_ped_zone import (
    ActorMachine,
    NPersonPedZoneMachine,
)
from core.structs.actor import ActorCategory
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class NoPedZoneTest(unittest.TestCase):
    def test_transitions(self) -> None:
        """Test that the incident machine transitions work as expected"""
        # Test update_zone_state trigger behavior
        actor_info = RelevantActorInfo(actor_id=101, track_uuid="101")
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")

        # update state with 1 person in ped zone
        machine.update_zone_state(
            num_persons_in_no_ped_zone=1,
        )
        self.assertEqual(machine.state, "start")

        # update state with 2 persons in ped zone
        machine.update_zone_state(
            num_persons_in_no_ped_zone=2,
        )
        self.assertEqual(machine.state, "more_persons_than_threshold")

        # update state with 1 persons in ped zone
        machine.update_zone_state(
            num_persons_in_no_ped_zone=1,
        )
        self.assertEqual(machine.state, "start")

        # update state with 2 persons in ped zone and over time thresh.
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 130_000
        machine.update_zone_state(
            num_persons_in_no_ped_zone=2,
        )
        machine.check_incident(incident_list=[])
        self.assertEqual(machine.state, "incident")

        # update state with 1 persons in ped zone
        machine.update_zone_state(
            num_persons_in_no_ped_zone=1,
        )
        self.assertEqual(machine.state, "start")

    def test_state_and_events(self):
        """Tests handling of state and event messages with no actor supressions."""

        # Test starting state
        machine = NPersonPedZoneMachine("", {})
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            num_persons_in_no_ped_zone=1,
            actor_category=ActorCategory.NO_PED_ZONE,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")

        # Test adding another actor
        _state = State(
            timestamp_ms=150_001,
            end_timestamp_ms=160_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            num_persons_in_no_ped_zone=2,
            actor_category=ActorCategory.NO_PED_ZONE,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(
            machine.actor_machines[track_uuid].state,
            "more_persons_than_threshold",
        )

        # Test going over time threshold
        _state = State(
            timestamp_ms=160_001,
            end_timestamp_ms=160_002,
            actor_id=actor_id,
            track_uuid=track_uuid,
            num_persons_in_no_ped_zone=2,
            actor_category=ActorCategory.NO_PED_ZONE,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(
            machine.actor_machines[track_uuid].state,
            "more_persons_than_threshold",
        )

        # Test generating a cooldown incident
        _state = State(
            timestamp_ms=160_000,
            end_timestamp_ms=170_004,
            actor_id=actor_id,
            track_uuid=track_uuid,
            num_persons_in_no_ped_zone=2,
            actor_category=ActorCategory.NO_PED_ZONE,
            camera_uuid=None,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(
            machine.actor_machines[track_uuid].state,
            "more_persons_than_threshold",
        )
        self.assertEqual(len(incidents), 1)
        incident = incidents[0]
        self.assertEqual(incident.cooldown_tag, True)

        # Test returning to start
        _state = State(
            timestamp_ms=170_004,
            end_timestamp_ms=190_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            num_persons_in_no_ped_zone=1,
            actor_category=ActorCategory.NO_PED_ZONE,
            camera_uuid=None,
        )
        machine.process_state_event([_state])
        self.assertEqual(
            machine.actor_machines[track_uuid].state,
            "start",
        )


if __name__ == "__main__":
    unittest.main()
