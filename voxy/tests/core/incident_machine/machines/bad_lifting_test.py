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

from core.incident_machine.machines.bad_lifting import (
    ActorMachine,
    BadLiftingViolationMachine,
)
from core.incident_machine.machines.base import RelevantActorInfo
from core.structs.actor import ActorCategory
from core.structs.state import PostureType, State


# trunk-ignore-all(pylint/E1101,pylint/R0915)
class BadLiftingTest(unittest.TestCase):
    def test_transitions(self) -> None:
        """Test that the incident machine transitions work as expected"""
        actor_info = RelevantActorInfo(
            actor_id=101,
            track_uuid="101",
        )
        # Test update_person_state trigger behavior, without carry classifier
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 6000
        machine.update_person_state(
            person_lift_type=PostureType.BAD, person_is_carrying_object=None
        )
        self.assertEqual(machine.state, "lifting_badly")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")

        # Test update_person_state trigger behavior, with carry
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 6000
        machine.update_person_state(
            person_lift_type=PostureType.BAD, person_is_carrying_object=True
        )
        self.assertEqual(machine.state, "lifting_badly")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")

        # Test update_person_state trigger behavior, without carry
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 6000
        machine.update_person_state(
            person_lift_type=PostureType.BAD,
            person_is_carrying_object=False,
        )
        self.assertEqual(machine.state, "start")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 0)
        self.assertEqual(machine.state, "start")

        # Test update_person_state trigger behavior
        machine = ActorMachine("", actor_info)
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 300
        machine.update_person_state(
            person_lift_type=PostureType.BAD,
            person_is_carrying_object=None,
        )
        self.assertEqual(machine.state, "lifting_badly")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")
        machine.state_event_start_timestamp_ms = 1000
        machine.state_event_end_timestamp_ms = 7000
        machine.update_person_state(
            person_lift_type=PostureType.GOOD,
            person_is_carrying_object=None,
        )
        self.assertEqual(machine.state, "start")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 0)
        self.assertEqual(machine.state, "start")

        # Test update_person_state trigger behavior with actor supressions removed
        machine = ActorMachine("", actor_info)
        machine.should_not_apply_actor_supressions = True
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 100
        machine.state_event_end_timestamp_ms = 300
        machine.update_person_state(
            person_lift_type=PostureType.BAD,
            person_is_carrying_object=None,
        )
        self.assertEqual(machine.state, "lifting_badly")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")
        machine.try_reset_state()
        self.assertEqual(machine.state, "start")
        machine.state_event_start_timestamp_ms = 1000
        machine.state_event_end_timestamp_ms = 7000
        machine.update_person_state(
            person_lift_type=PostureType.BAD,
            person_is_carrying_object=None,
        )
        self.assertEqual(machine.state, "lifting_badly")
        incident_list = []
        machine.check_incident(incident_list=incident_list)
        self.assertEqual(len(incident_list), 1)
        self.assertEqual(machine.state, "incident")

    def test_state_and_events(self):
        """Tests handling of state and event messages with no actor supressions."""

        # Test without carry classifier
        machine = BadLiftingViolationMachine("", {})
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            person_lift_type=PostureType.GOOD,
            actor_category=ActorCategory.PERSON,
            camera_uuid=None,
            track_uuid=track_uuid,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 0)

        _state = State(
            timestamp_ms=150_000,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            person_lift_type=PostureType.BAD,
            actor_category=ActorCategory.PERSON,
            camera_uuid=None,
            track_uuid=track_uuid,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 1)

        _state = State(
            timestamp_ms=150_100,
            end_timestamp_ms=350_000,
            actor_id=actor_id,
            person_lift_type=PostureType.BAD,
            actor_category=ActorCategory.PERSON,
            camera_uuid=None,
            track_uuid=track_uuid,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 1)
        incident = incidents[0]
        self.assertEqual(incident.cooldown_tag, True)

        # Test incident generation with carry classifier as True
        machine = BadLiftingViolationMachine("", {})
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            person_lift_type=PostureType.GOOD,
            actor_category=ActorCategory.PERSON,
            person_is_carrying_object=False,
            camera_uuid=None,
            track_uuid=track_uuid,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 0)

        _state = State(
            timestamp_ms=150_000,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            person_lift_type=PostureType.BAD,
            actor_category=ActorCategory.PERSON,
            person_is_carrying_object=True,
            camera_uuid=None,
            track_uuid=track_uuid,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 1)

        # Test incident generation with carry classifier as False
        machine = BadLiftingViolationMachine("", {})
        actor_id = 1
        track_uuid = "1"
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            person_lift_type=PostureType.GOOD,
            actor_category=ActorCategory.PERSON,
            person_is_carrying_object=False,
            camera_uuid=None,
            track_uuid=track_uuid,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 0)

        _state = State(
            timestamp_ms=150_000,
            end_timestamp_ms=150_000,
            actor_id=actor_id,
            person_lift_type=PostureType.BAD,
            actor_category=ActorCategory.PERSON,
            person_is_carrying_object=False,
            camera_uuid=None,
            track_uuid=track_uuid,
        )
        incidents = machine.process_state_event([_state])
        self.assertEqual(machine.actor_machines[track_uuid].state, "start")
        self.assertEqual(len(incidents), 0)


if __name__ == "__main__":
    unittest.main()
