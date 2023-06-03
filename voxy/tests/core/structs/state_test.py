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

from core.structs.actor import ActorCategory, DoorState
from core.structs.ergonomics import PostureType

# trunk-ignore(pylint/E0611)
from core.structs.protobufs.v1.state_pb2 import State as StatePb
from core.structs.state import State

# trunk-ignore-all(pylint/C0116)


class StateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = State(
            1,
            "other_uuid",
            "1",
            ActorCategory.PIT,
            1,
            "uuid",
            False,
            DoorState.PARTIALLY_OPEN,
            0,
            PostureType.GOOD,
            PostureType.BAD,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )

    def test_to_proto(self) -> None:
        self.assertTrue(self.state.to_proto() is not None)
        self.assertTrue(isinstance(self.state.to_proto(), StatePb))

    def test_grouping_key(self) -> None:
        self.assertTrue(self.state.grouping_key is not None)

    def test_differentiator(self) -> None:
        self.state.actor_category = ActorCategory.PIT
        self.assertTrue(self.state.differentiator is not None)
        self.state.actor_category = ActorCategory.PERSON
        self.assertTrue(self.state.differentiator is not None)
        self.state.actor_category = ActorCategory.DOOR
        self.assertTrue(self.state.differentiator is not None)
        self.state.actor_category = ActorCategory.MOTION_DETECTION_ZONE
        self.assertTrue(self.state.differentiator is not None)
        self.state.actor_category = None
        self.assertTrue(self.state.differentiator is None)

    def test_proto_roundtrip(self) -> None:
        states = [
            State(
                timestamp_ms=10,
                camera_uuid=str(uuid.uuid4()),
                actor_id=str(uuid.uuid4()),
                actor_category=ActorCategory.BIKE,
                end_timestamp_ms=20,
                run_uuid=str(uuid.uuid4()),
                door_is_open=True,
                door_state=DoorState.PARTIALLY_OPEN,
                motion_zone_is_in_motion=True,
                person_lift_type=PostureType.BAD,
                person_reach_type=PostureType.GOOD,
                person_is_wearing_safety_vest=True,
                person_is_wearing_hard_hat=True,
                person_is_carrying_object=True,
                pit_is_stationary=False,
                person_is_associated=True,
                person_in_no_ped_zone=True,
                pit_in_driving_area=True,
                pit_is_associated=True,
                num_persons_in_no_ped_zone=5,
                track_uuid=str(uuid.uuid4()),
            ),
            State(
                timestamp_ms=10,
                camera_uuid=str(uuid.uuid4()),
                actor_id=str(uuid.uuid4()),
                actor_category=ActorCategory.BIKE,
                end_timestamp_ms=20,
                run_uuid=str(uuid.uuid4()),
                door_is_open=True,
                door_state=DoorState.PARTIALLY_OPEN,
                motion_zone_is_in_motion=True,
                person_lift_type=PostureType.BAD,
                person_reach_type=None,
                person_is_wearing_safety_vest=None,
                person_is_wearing_hard_hat=None,
                person_is_carrying_object=None,
                pit_is_stationary=False,
                person_is_associated=True,
                person_in_no_ped_zone=True,
                pit_in_driving_area=True,
                pit_is_associated=True,
                num_persons_in_no_ped_zone=5,
                track_uuid=str(uuid.uuid4()),
            ),
        ]

        for state in states:
            self.assertEqual(state, State.from_proto(state.to_proto()))


if __name__ == "__main__":
    unittest.main()
