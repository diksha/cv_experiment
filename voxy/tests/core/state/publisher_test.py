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

from core.state.publisher import Publisher
from core.structs.actor import ActorCategory
from core.structs.event import Event
from core.structs.state import State


class PublisherTest(unittest.TestCase):
    def test_conditional_google_pub_sub(self) -> None:
        """
        Tests pubsub to see if it only publishes motion detection zones
        """

        def make_fake_state_message(
            actor_category_value: ActorCategory,
        ) -> State:
            """
            Generates a dummy state message with the actor category

            Args:
                actor_category_value(ActorCategory): the actor category

            Returns:
                State: the state message
            """
            return State(
                actor_category=actor_category_value,
                timestamp_ms=0,
                camera_uuid="foo",
                actor_id=1,
            )

        self.assertTrue(Publisher is not None)
        # Production line down incident
        motion_zone_state = make_fake_state_message(
            ActorCategory.MOTION_DETECTION_ZONE
        )
        motion_zone_event = Event(0, "foo", "1", "NOT_MOTION", "2")
        self.assertTrue(
            Publisher.should_generate_google_pubsub_message(motion_zone_state)
        )
        self.assertFalse(
            Publisher.should_generate_google_pubsub_message(motion_zone_event)
        )

        non_motion_zone_actors = [
            ActorCategory.PIT,
            ActorCategory.DOOR,
            ActorCategory.HARD_HAT,
            ActorCategory.SAFETY_VEST,
            ActorCategory.BARE_CHEST,
            ActorCategory.BARE_HEAD,
            ActorCategory.INTERSECTION,
            ActorCategory.AISLE_END,
            ActorCategory.PERSON_V2,
            ActorCategory.PIT_V2,
            ActorCategory.NO_PED_ZONE,
            ActorCategory.DRIVING_AREA,
            ActorCategory.TRUCK,
            ActorCategory.VEHICLE,
            ActorCategory.TRAILER,
            ActorCategory.BIKE,
            ActorCategory.BUS,
            ActorCategory.SAFETY_GLOVE,
            ActorCategory.BARE_HAND,
            ActorCategory.SPILL,
        ]
        for actor_category in non_motion_zone_actors:
            non_motion_zone_message = make_fake_state_message(actor_category)
            self.assertFalse(
                Publisher.should_generate_google_pubsub_message(
                    non_motion_zone_message
                )
            )


if __name__ == "__main__":
    unittest.main()
