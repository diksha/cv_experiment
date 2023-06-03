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

import numpy as np

from core.perception.acausal.algorithms.track_smoothing import (
    TrackSmoothingAlgorithm,
)
from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Point, Polygon
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class TrackletTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up test...")

        self.vignette = Vignette()

        # Set up tracklet
        self.tracklet = Tracklet()
        self.tracklet.category = ActorCategory.PIT

        # Add to tracklet
        self.tracklet.update(
            Actor(
                category=ActorCategory.PIT,
                polygon=Polygon(
                    [
                        Point(x=5, y=15, z=None),
                        Point(x=5, y=25, z=None),
                        Point(x=15, y=15, z=None),
                        Point(x=15, y=25, z=None),
                    ]
                ),
            ),
            timestamp_ms=0,
        )

        self.tracklet.update(
            Actor(
                category=ActorCategory.PIT,
                polygon=Polygon(
                    [
                        Point(x=15, y=15, z=None),
                        Point(x=15, y=25, z=None),
                        Point(x=25, y=15, z=None),
                        Point(x=25, y=25, z=None),
                    ]
                ),
            ),
            timestamp_ms=200,
        )

        self.vignette.tracklets[0] = self.tracklet

    def test_process_vignette(self) -> None:
        smoother = TrackSmoothingAlgorithm({})

        tracklet = self.vignette.tracklets[0]

        before_smoothing = tracklet.xysr_track
        self.vignette = smoother.process_vignette(self.vignette)
        after_smoothing = self.vignette.tracklets[0].xysr_track

        print(
            f"Before smoothing: {before_smoothing},{before_smoothing.shape} \nAfter smoothing: {after_smoothing}"
        )

        # Should be the same shape before and after smoothing
        self.assertEqual(
            before_smoothing.shape,
            after_smoothing.shape,
        )

        # Should not be the same after smoothing
        self.assertFalse(np.array_equal(before_smoothing, after_smoothing))

        # Check that the values are all the same
        for i, timestamp in enumerate(tracklet.timestamps):
            print(f"i: {i}\nTimestamp: {timestamp}")
            print(
                f"actor: {tracklet[tracklet.get_actor_index_at_time(timestamp)]}"
            )

            actor = tracklet[
                tracklet.get_actor_index_at_time(
                    timestamp, allow_closest_earlier_timestamp=False
                )
            ]  # this implicitly checks that the timestamps are the same

            # Check that polygons are the same
            # trunk-ignore(pylint/W0212)
            actor_box = tracklet._convert_bbox_to_xysr(actor.polygon).squeeze()
            print(
                f"1: {tracklet.xysr(i)}\n2: {actor_box}, {tracklet.xysr(i).shape}, {actor_box.shape}"
            )
            self.assertTrue(
                np.array_equal(
                    tracklet.xysr(i),
                    actor_box,
                )
            )


if __name__ == "__main__":
    unittest.main()
