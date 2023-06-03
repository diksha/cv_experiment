#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import logging
import unittest

from core.perception.acausal.algorithms.association_controller import (
    AssociationAlgorithmController,
)
from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Point, Polygon
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette

K_PERSON_TRACK_ID_1 = 100
K_PIT_TRACK_ID_1 = 101
K_PERSON_TRACK_ID_2 = 102
K_PIT_TRACK_ID_2 = 103
K_LOOKBACK_TIME_MS_TEST = 5000
K_MIN_FRACTION_TO_ASSOCIATE_TEST = 0.5


class PitPersonTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up test...")

        self.current_time_ms = 100

        self.vignette = Vignette()
        self.vignette.present_timestamp_ms = self.current_time_ms

        # Set up PERSON tracklet
        self.person_tracklet_1 = Tracklet()
        self.person_tracklet_1.category = ActorCategory.PERSON
        self.person_tracklet_1.track_id = K_PERSON_TRACK_ID_1

        self.person_tracklet_2 = Tracklet()
        self.person_tracklet_2.category = ActorCategory.PERSON
        self.person_tracklet_2.track_id = K_PERSON_TRACK_ID_1

        # Set up PIT tracklet
        self.pit_tracklet_1 = Tracklet()
        self.pit_tracklet_1.category = ActorCategory.PIT
        self.pit_tracklet_1.track_id = K_PIT_TRACK_ID_1

        self.pit_tracklet_2 = Tracklet()
        self.pit_tracklet_2.category = ActorCategory.PIT
        self.pit_tracklet_2.track_id = K_PIT_TRACK_ID_2

        # Add to tracklet
        self.person_tracklet_1.update(
            Actor(
                category=ActorCategory.PERSON,
                polygon=Polygon(
                    [
                        Point(x=0, y=0, z=None),
                        Point(x=0, y=5, z=None),
                        Point(x=5, y=0, z=None),
                        Point(x=5, y=5, z=None),
                    ]
                ),
                track_id=K_PERSON_TRACK_ID_1,
            ),
            timestamp_ms=self.current_time_ms,
        )

        self.person_tracklet_2.update(
            Actor(
                category=ActorCategory.PERSON,
                polygon=Polygon(
                    [
                        Point(x=20, y=0, z=None),
                        Point(x=20, y=5, z=None),
                        Point(x=25, y=0, z=None),
                        Point(x=25, y=5, z=None),
                    ]
                ),
                track_id=K_PERSON_TRACK_ID_2,
            ),
            timestamp_ms=self.current_time_ms,
        )

        self.pit_tracklet_1.update(
            Actor(
                category=ActorCategory.PIT,
                polygon=Polygon(
                    [
                        Point(x=0, y=0, z=None),
                        Point(x=0, y=5, z=None),
                        Point(x=5, y=0, z=None),
                        Point(x=5, y=5, z=None),
                    ]
                ),
                track_id=K_PIT_TRACK_ID_1,
            ),
            timestamp_ms=self.current_time_ms,
        )

        self.pit_tracklet_2.update(
            Actor(
                category=ActorCategory.PIT,
                polygon=Polygon(
                    [
                        Point(x=20, y=0, z=None),
                        Point(x=20, y=5, z=None),
                        Point(x=25, y=0, z=None),
                        Point(x=25, y=5, z=None),
                    ]
                ),
                track_id=K_PIT_TRACK_ID_2,
            ),
            timestamp_ms=self.current_time_ms,
        )

        self.vignette.tracklets[K_PERSON_TRACK_ID_1] = self.person_tracklet_1
        self.vignette.tracklets[K_PERSON_TRACK_ID_2] = self.person_tracklet_2
        self.vignette.tracklets[K_PIT_TRACK_ID_1] = self.pit_tracklet_1
        self.vignette.tracklets[K_PIT_TRACK_ID_2] = self.pit_tracklet_2

    def _update_time_100_ms(self) -> None:
        self.current_time_ms += 100
        self.vignette.present_timestamp_ms = self.current_time_ms

    def _dissasociate_person_1_pit_1(self) -> None:
        self.person_tracklet_1.update(
            Actor(
                category=ActorCategory.PERSON,
                polygon=Polygon(
                    [
                        Point(x=100, y=100, z=None),
                        Point(x=100, y=105, z=None),
                        Point(x=105, y=100, z=None),
                        Point(x=105, y=105, z=None),
                    ]
                ),
                track_id=K_PERSON_TRACK_ID_1,
            ),
            timestamp_ms=self.current_time_ms,
        )

        self.person_tracklet_2.update(
            self.person_tracklet_2[-1],
            timestamp_ms=self.current_time_ms,
        )

        self.pit_tracklet_1.update(
            self.pit_tracklet_1[-1],
            timestamp_ms=self.current_time_ms,
        )

        self.pit_tracklet_2.update(
            self.pit_tracklet_2[-1], timestamp_ms=self.current_time_ms
        )

    def _associate_person_2_pit_1(self) -> None:
        self.person_tracklet_1.update(
            self.person_tracklet_1[-1],
            timestamp_ms=self.current_time_ms,
        )

        self.person_tracklet_2.update(
            Actor(
                category=ActorCategory.PERSON,
                polygon=Polygon(
                    [
                        Point(x=0, y=0, z=None),
                        Point(x=0, y=5, z=None),
                        Point(x=5, y=0, z=None),
                        Point(x=5, y=5, z=None),
                    ]
                ),
                track_id=K_PERSON_TRACK_ID_2,
            ),
            timestamp_ms=self.current_time_ms,
        )

        self.pit_tracklet_1.update(
            self.pit_tracklet_1[-1],
            timestamp_ms=self.current_time_ms,
        )

        self.pit_tracklet_2.update(
            self.pit_tracklet_2[-1],
            timestamp_ms=self.current_time_ms,
        )

    def _overlap_people_and_pits_with_offsets(self) -> None:
        self.person_tracklet_1.update(
            self.person_tracklet_1[-1],
            timestamp_ms=self.current_time_ms,
        )

        self.person_tracklet_2.update(
            Actor(
                category=ActorCategory.PERSON,
                polygon=Polygon(
                    [
                        Point(x=1, y=0, z=None),
                        Point(x=1, y=5, z=None),
                        Point(x=6, y=0, z=None),
                        Point(x=6, y=5, z=None),
                    ]
                ),
                track_id=K_PERSON_TRACK_ID_2,
            ),
            timestamp_ms=self.current_time_ms,
        )

        self.pit_tracklet_1.update(
            Actor(
                category=ActorCategory.PIT,
                polygon=Polygon(
                    [
                        Point(x=-1, y=0, z=None),
                        Point(x=-1, y=5, z=None),
                        Point(x=4, y=0, z=None),
                        Point(x=4, y=5, z=None),
                    ]
                ),
                track_id=K_PIT_TRACK_ID_1,
            ),
            timestamp_ms=self.current_time_ms,
        )

        self.pit_tracklet_2.update(
            Actor(
                category=ActorCategory.PIT,
                polygon=Polygon(
                    [
                        Point(x=0, y=0, z=None),
                        Point(x=0, y=5, z=None),
                        Point(x=5, y=0, z=None),
                        Point(x=5, y=5, z=None),
                    ]
                ),
                track_id=K_PIT_TRACK_ID_2,
            ),
            timestamp_ms=self.current_time_ms,
        )

    def test_initial_association(self) -> None:
        controller = AssociationAlgorithmController(
            {"logging_level": logging.DEBUG}
        )
        controller.process_vignette(self.vignette)

        # PERSON 1
        tracklet = self.vignette.tracklets[K_PERSON_TRACK_ID_1]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PIT, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PIT
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PIT,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PIT_TRACK_ID_1)
        self.assertTrue(smoothed_association == K_PIT_TRACK_ID_1)
        self.assertTrue(len(temporal_association) == 1)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertTrue(tracklet.is_associated_with_pit)

        # PERSON 2
        tracklet = self.vignette.tracklets[K_PERSON_TRACK_ID_2]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PIT, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PIT
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PIT,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PIT_TRACK_ID_2)
        self.assertTrue(smoothed_association == K_PIT_TRACK_ID_2)
        self.assertTrue(len(temporal_association) == 1)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertTrue(tracklet.is_associated_with_pit)

        # PIT 1
        tracklet = self.vignette.tracklets[K_PIT_TRACK_ID_1]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PERSON, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PERSON
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PERSON,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PERSON_TRACK_ID_1)
        self.assertTrue(smoothed_association == K_PERSON_TRACK_ID_1)
        self.assertTrue(len(temporal_association) == 1)
        self.assertTrue(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)

        # PIT 2
        tracklet = self.vignette.tracklets[K_PIT_TRACK_ID_2]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PERSON, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PERSON
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PERSON,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PERSON_TRACK_ID_2)
        self.assertTrue(smoothed_association == K_PERSON_TRACK_ID_2)
        self.assertTrue(len(temporal_association) == 1)
        self.assertTrue(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)

    def test_disassociate_person_1(self) -> None:
        controller = AssociationAlgorithmController(
            {"logging_level": logging.DEBUG}
        )
        controller.process_vignette(self.vignette)
        self._update_time_100_ms()
        self._dissasociate_person_1_pit_1()
        controller.process_vignette(self.vignette)

        # PERSON 1: NOT ASSOCIATED
        tracklet = self.vignette.tracklets[K_PERSON_TRACK_ID_1]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PIT, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PIT
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PIT,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertFalse(raw_association == K_PIT_TRACK_ID_1)
        self.assertFalse(smoothed_association == K_PIT_TRACK_ID_1)
        self.assertTrue(len(temporal_association) == 2)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)

        # PERSON 2: NO CHANGE
        tracklet = self.vignette.tracklets[K_PERSON_TRACK_ID_2]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PIT, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PIT
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PIT,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PIT_TRACK_ID_2)
        self.assertTrue(smoothed_association == K_PIT_TRACK_ID_2)
        self.assertTrue(len(temporal_association) == 2)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertTrue(tracklet.is_associated_with_pit)

        # PIT 1: NOT ASSOCIATED
        tracklet = self.vignette.tracklets[K_PIT_TRACK_ID_1]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PERSON, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PERSON
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PERSON,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertFalse(raw_association == K_PERSON_TRACK_ID_1)
        self.assertFalse(smoothed_association == K_PERSON_TRACK_ID_1)
        self.assertTrue(len(temporal_association) == 2)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)

        # PIT 2: NO CHANGE
        tracklet = self.vignette.tracklets[K_PIT_TRACK_ID_2]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PERSON, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PERSON
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PERSON,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PERSON_TRACK_ID_2)
        self.assertTrue(smoothed_association == K_PERSON_TRACK_ID_2)
        self.assertTrue(len(temporal_association) == 2)
        self.assertTrue(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)

    def test_disassociate_person_1_associate_person_2(self) -> None:
        controller = AssociationAlgorithmController(
            {"logging_level": logging.DEBUG}
        )
        controller.process_vignette(self.vignette)
        self._update_time_100_ms()
        self._dissasociate_person_1_pit_1()
        controller.process_vignette(self.vignette)
        self._update_time_100_ms()
        self._associate_person_2_pit_1()
        controller.process_vignette(self.vignette)

        # PERSON 1: NOT ASSOCIATED
        tracklet = self.vignette.tracklets[K_PERSON_TRACK_ID_1]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PIT, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PIT
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PIT,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertFalse(raw_association == K_PIT_TRACK_ID_1)
        self.assertFalse(smoothed_association == K_PIT_TRACK_ID_1)
        self.assertTrue(len(temporal_association) == 3)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)

        # PERSON 2: RAW ASSOCIATED WITH PIT 1 BUT SMOOTHED
        # ASSOCIATED WITH PIT 2
        tracklet = self.vignette.tracklets[K_PERSON_TRACK_ID_2]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PIT, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PIT
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PIT,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PIT_TRACK_ID_1)
        self.assertTrue(smoothed_association == K_PIT_TRACK_ID_2)
        self.assertTrue(len(temporal_association) == 3)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertTrue(tracklet.is_associated_with_pit)

        # PIT 1: RAW ASSOCIATED WITH PERSON 2 BUT STILL NOT
        # SMOOTHED ASSOCIATED
        tracklet = self.vignette.tracklets[K_PIT_TRACK_ID_1]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PERSON, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PERSON
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PERSON,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PERSON_TRACK_ID_2)
        self.assertFalse(smoothed_association == K_PERSON_TRACK_ID_1)
        self.assertTrue(len(temporal_association) == 3)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)

        # PIT 2: NO LONGER RAW ASSOCIATED BUT SMOOTHED ASSOCIATED
        # WITH PERSON 2
        tracklet = self.vignette.tracklets[K_PIT_TRACK_ID_2]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PERSON, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PERSON
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PERSON,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertFalse(raw_association == K_PERSON_TRACK_ID_2)
        self.assertTrue(smoothed_association == K_PERSON_TRACK_ID_2)
        self.assertTrue(len(temporal_association) == 3)
        self.assertTrue(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)

    def test_hungarian(self) -> None:
        controller = AssociationAlgorithmController(
            {"logging_level": logging.DEBUG}
        )
        controller.process_vignette(self.vignette)
        self._update_time_100_ms()
        self._overlap_people_and_pits_with_offsets()
        controller.process_vignette(self.vignette)

        # PERSON 1
        tracklet = self.vignette.tracklets[K_PERSON_TRACK_ID_1]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PIT, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PIT
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PIT,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PIT_TRACK_ID_1)
        self.assertTrue(smoothed_association == K_PIT_TRACK_ID_1)
        self.assertTrue(len(temporal_association) == 2)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertTrue(tracklet.is_associated_with_pit)

        # PERSON 2
        tracklet = self.vignette.tracklets[K_PERSON_TRACK_ID_2]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PIT, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PIT
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PIT,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PIT_TRACK_ID_2)
        self.assertTrue(smoothed_association == K_PIT_TRACK_ID_2)
        self.assertTrue(len(temporal_association) == 2)
        self.assertFalse(tracklet.is_associated_with_person)
        self.assertTrue(tracklet.is_associated_with_pit)

        # PIT 1
        tracklet = self.vignette.tracklets[K_PIT_TRACK_ID_1]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PERSON, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PERSON
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PERSON,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PERSON_TRACK_ID_1)
        self.assertTrue(smoothed_association == K_PERSON_TRACK_ID_1)
        self.assertTrue(len(temporal_association) == 2)
        self.assertTrue(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)

        # PIT 2
        tracklet = self.vignette.tracklets[K_PIT_TRACK_ID_2]
        raw_association = tracklet.get_raw_associated_tracklet_at_time(
            ActorCategory.PERSON, self.vignette.present_timestamp_ms
        )
        temporal_association = tracklet.get_temporal_association_for_actor(
            ActorCategory.PERSON
        )
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PERSON,
            self.vignette.present_timestamp_ms,
            K_LOOKBACK_TIME_MS_TEST,
            K_MIN_FRACTION_TO_ASSOCIATE_TEST,
        )
        self.assertTrue(raw_association == K_PERSON_TRACK_ID_2)
        self.assertTrue(smoothed_association == K_PERSON_TRACK_ID_2)
        self.assertTrue(len(temporal_association) == 2)
        self.assertTrue(tracklet.is_associated_with_person)
        self.assertFalse(tracklet.is_associated_with_pit)


if __name__ == "__main__":
    unittest.main()
