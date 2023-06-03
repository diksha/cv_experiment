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
import unittest

from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Polygon
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette

TRACK_ID_1 = 101
TRACK_ID_2 = 102
TRACK_ID_3 = 103
TRACK_ID_4 = 104


class VignetteTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up test...")
        time_ms = 0

        # Set up vignette
        self.vignette = Vignette()
        self.vignette.present_timestamp_ms = time_ms

        # Create Tracklet 1
        self.tracklet_1 = Tracklet()
        self.tracklet_1.track_id = TRACK_ID_1
        self.tracklet_1.update(
            Actor(
                category=ActorCategory.PIT,
                polygon=Polygon.from_xysr([0, 5, 100, 1]),
            ),
            timestamp_ms=time_ms,
        )

        self.tracklet_2 = Tracklet()
        self.tracklet_2.track_id = TRACK_ID_2
        self.tracklet_2.update(
            Actor(
                category=ActorCategory.PIT,
                polygon=Polygon.from_xysr([10, 20, 100, 1]),
            ),
            timestamp_ms=time_ms,
        )

        self.tracklet_3 = Tracklet()
        self.tracklet_3.track_id = TRACK_ID_3
        self.tracklet_3.update(
            Actor(
                category=ActorCategory.PERSON,
                polygon=Polygon.from_xysr([10, 5, 100, 1]),
            ),
            timestamp_ms=time_ms,
        )

        self.tracklet_4 = Tracklet()
        self.tracklet_4.track_id = TRACK_ID_4
        self.tracklet_4.update(
            Actor(
                category=ActorCategory.PERSON,
                polygon=Polygon.from_xysr([30, 10, 100, 1]),
            ),
            timestamp_ms=time_ms,
        )

        self.vignette.tracklets[TRACK_ID_1] = self.tracklet_1
        self.vignette.tracklets[TRACK_ID_2] = self.tracklet_2
        self.vignette.tracklets[TRACK_ID_3] = self.tracklet_3
        self.vignette.tracklets[TRACK_ID_4] = self.tracklet_4

    def test_l2_distance(self) -> None:
        filtered_tracklets = self.vignette.filter_null_xysr_tracks(
            ActorCategory.PERSON_V2
        )
        self.assertEqual(len(filtered_tracklets), 2)
        self.assertEqual(len(filtered_tracklets[0]), 0)

        filtered_tracklets = self.vignette.filter_null_xysr_tracks(
            ActorCategory.PIT
        )
        self.assertEqual(len(filtered_tracklets), 2)
        self.assertEqual(len(filtered_tracklets[0]), 2)


if __name__ == "__main__":
    unittest.main()
