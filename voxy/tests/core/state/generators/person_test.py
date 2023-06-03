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

from core.state.generators.person import PersonStateGenerator
from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Point, Polygon
from core.structs.ergonomics import Activity, ActivityType, PostureType
from core.structs.frame import Frame
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class PersonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.person_state_generator = PersonStateGenerator(
            {"camera_uuid": "uscold/laredo/dock03/cha"}
        )
        self.tracklet = Tracklet()

    def testGenerator(self) -> None:

        # test empty vignette
        present_frame_struct = Frame(
            frame_number=0,
            frame_width=1280,
            frame_height=720,
            relative_timestamp_s=0,
            relative_timestamp_ms=0,
            epoch_timestamp_ms=0,
        )
        vignette = Vignette(present_frame_struct=present_frame_struct)

        # Test empty vignette
        generator_response = self.person_state_generator.process_vignette(
            vignette
        )
        self.assertFalse(generator_response.states)  # expecting empty states
        self.assertFalse(generator_response.events)  # expecting empty events

        # Test lifting/reaching actor

        actor = Actor(
            track_id=1,
            category=ActorCategory.PERSON,
            polygon=Polygon(
                [
                    Point(x=5, y=15, z=None),
                    Point(x=5, y=25, z=None),
                    Point(x=15, y=15, z=None),
                    Point(x=15, y=25, z=None),
                ]
            ),
            activity={
                "LIFTING": Activity(
                    ActivityType(ActivityType.LIFTING),
                    PostureType(PostureType.GOOD.value),
                ),
                "REACHING": Activity(
                    ActivityType(ActivityType.REACHING),
                    PostureType(PostureType.BAD.value),
                ),
            },
        )

        # Add to tracklet
        self.tracklet.update(
            instance=actor,
            timestamp_ms=0,
        )
        self.tracklet.is_believed_to_be_in_unsafe_posture = True
        present_frame_struct = Frame(
            frame_number=0,
            frame_width=1280,
            frame_height=720,
            relative_timestamp_s=0,
            relative_timestamp_ms=0,
            epoch_timestamp_ms=0,
            actors=[actor],
        )

        vignette = Vignette(
            tracklets={actor.track_id: self.tracklet},
            present_frame_struct=present_frame_struct,
            present_timestamp_ms=0,
        )
        generator_response = self.person_state_generator.process_vignette(
            vignette
        )
        self.assertEqual(
            generator_response.states[0].person_lift_type.name, "GOOD"
        )
        self.assertEqual(
            generator_response.states[0].person_reach_type.name, "BAD"
        )


if __name__ == "__main__":
    unittest.main()
