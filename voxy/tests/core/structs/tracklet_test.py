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
import timeit
import unittest
from unittest.mock import patch

import numpy as np

from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Point, Polygon
from core.structs.tracklet import Tracklet

# trunk can't see these protos
# trunk-ignore-begin(pylint/E0611)
from protos.perception.structs.v1.tracklet_pb2 import Tracklet as TrackletPb

# trunk-ignore-end(pylint/E0611)


class TrackletTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up test...")

        # Set up tracklet
        self.tracklet = Tracklet()

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

        self.tracklet.update(None, timestamp_ms=100)

    def test_get_actor_index_at_time(self) -> None:
        # allow closest false, actor instance exists
        self.assertEqual(
            self.tracklet.get_actor_index_at_time(
                200, allow_closest_earlier_timestamp=False
            ),
            2,
        )

        # allow closest false, actor instance does not exist
        self.assertIsNone(
            self.tracklet.get_actor_index_at_time(
                199, allow_closest_earlier_timestamp=False
            )
        )

        # allow closest true, actor instance exists
        self.assertEqual(
            self.tracklet.get_actor_index_at_time(
                200, allow_closest_earlier_timestamp=True
            ),
            2,
        )

        # allow closest true, actor instance does not exist
        self.assertEqual(
            self.tracklet.get_actor_index_at_time(
                199, allow_closest_earlier_timestamp=True
            ),
            1,
        )

    def test_get_closest_non_null_actor_at_time(self) -> None:
        # Check left actor
        expected_index = 0
        index = self.tracklet.get_closest_non_null_actor_at_time(50)
        self.assertEqual(expected_index, index)

        # Check right actor & handling of null
        expected_index = 2
        index = self.tracklet.get_closest_non_null_actor_at_time(150)
        self.assertEqual(expected_index, index)

        # Check handling of null
        not_expected_index = 1
        index = self.tracklet.get_closest_non_null_actor_at_time(100)
        self.assertNotEqual(not_expected_index, index)

    def test_convert_xysr_to_bbox(self) -> None:
        xysr = [10, 20, 100, 1]
        bbox_expected = np.array([5, 15, 15, 25])

        # trunk-ignore(pylint/W0212)
        bbox = self.tracklet._convert_xysr_to_bbox(xysr)
        self.assertTrue(np.allclose(bbox, bbox_expected))

    def test_convert_bbox_to_xysr(self) -> None:
        xysr_expected = np.array([10, 20, 100, 1]).reshape((4, 1))

        polygon = Polygon(
            [
                Point(x=5, y=15, z=None),
                Point(x=5, y=25, z=None),
                Point(x=15, y=15, z=None),
                Point(x=15, y=25, z=None),
            ]
        )

        # trunk-ignore(pylint/W0212)
        xysr = self.tracklet._convert_bbox_to_xysr(polygon)

        self.assertTrue(np.allclose(xysr, xysr_expected))

    def test_get_xysr_at_time(self) -> None:
        # No interpolation, None
        empty_tracklet = Tracklet()
        xysr = empty_tracklet.get_xysr_at_time(timestamp_ms=0)
        self.assertIsNone(xysr)

        # No interpolation, at exact time
        xysr_expected = np.array([10, 20, 100, 1]).reshape((1, 4))
        xysr = self.tracklet.get_xysr_at_time(timestamp_ms=0)

        self.assertTrue(np.allclose(xysr, xysr_expected))

        # No interpolation, inbetween time
        xysr_expected = np.array([10, 20, 100, 1]).reshape((1, 4))
        xysr = self.tracklet.get_xysr_at_time(timestamp_ms=50)

        self.assertTrue(np.allclose(xysr, xysr_expected))

        # With interpolation
        xysr_expected = np.array([15, 20, 100, 1]).reshape((1, 4))
        xysr = self.tracklet.get_xysr_at_time(
            timestamp_ms=100, interpolate=True
        )

        self.assertTrue(np.allclose(xysr, xysr_expected))

        # No interpolation, null actor
        self.tracklet.update(
            None,
            timestamp_ms=250,
        )

        xysr_expected = np.array([20, 20, 100, 1]).reshape((1, 4))
        xysr = self.tracklet.get_xysr_at_time(
            timestamp_ms=250, interpolate=False
        )

        self.assertTrue(np.allclose(xysr, xysr_expected))

    def test_get_xysr_at_time_range(self) -> None:
        time_range = [50, 100, 150]

        xysr_expected = np.array(
            [
                [12.5, 20, 100, 1],
                [15, 20, 100, 1],
                [17.5, 20, 100, 1],
            ]
        ).reshape((-1, 4))

        xysr = self.tracklet.get_xysr_at_time_range(
            timestamp_range_ms=time_range
        )

        print("Expected and result: ", xysr, xysr_expected)
        self.assertTrue(np.allclose(xysr, xysr_expected))

    def test_get_actor_at_timestamp(self) -> None:
        self.assertTrue(self.tracklet.get_actor_at_timestamp(0.0) is not None)
        self.assertTrue(self.tracklet.get_actor_at_timestamp(-1) is None)

    def test_to_proto(self) -> None:
        proto = self.tracklet.to_proto()
        self.assertTrue(isinstance(proto, TrackletPb))

    def test_update(self) -> None:
        with patch(
            "core.structs.tracklet.MAX_TRACKLET_ACTOR_INSTANCES"
        ) as mock_get_item:
            mock_get_item.spec = int
            mock_get_item.__eq__ = lambda x, y: y == 5
            tracklet = Tracklet()
            self.assertTrue(mock_get_item == 5)
            self.assertTrue(5 == mock_get_item)
            tracklet.update(
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
            print(tracklet)
            self.assertTrue(len(tracklet.timestamps) == 1)
            self.assertTrue(tracklet.xysr_track.shape == (4, 1))
            tracklet.update(
                None,
                timestamp_ms=5,
            )
            self.assertTrue(len(tracklet.timestamps) == 1)
            self.assertTrue(tracklet.xysr_track.shape == (4, 1))
            time = 1
            for _ in range(100):
                tracklet.update(
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
                    timestamp_ms=time,
                )
                time += 1

            self.assertTrue(len(tracklet.timestamps) == 5)
            print(tracklet.xysr_track)
            print(tracklet.xysr_track.shape)
            self.assertTrue(tracklet.xysr_track.shape == (4, 5))
            tracklet.update(
                None,
                timestamp_ms=1000,
            )
            self.assertTrue(list(tracklet.timestamps) == [96, 97, 98, 99, 100])
            print(tracklet.xysr_track)
            self.assertTrue(tracklet.xysr_track.shape == (4, 5))
            tracklet.update(
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
                timestamp_ms=1001,
            )
            print(tracklet.timestamps)
            self.assertTrue(list(tracklet.timestamps) == [98, 99, 100, 1001])
            tracklet.update(
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
                timestamp_ms=1002,
            )
            print(tracklet.timestamps)
            self.assertTrue(list(tracklet.timestamps) == [99, 100, 1001, 1002])
            tracklet.update(
                Actor(
                    category=ActorCategory.PIT,
                    polygon=Polygon(
                        [
                            Point(x=5, y=15, z=None),
                            Point(x=5, y=25, z=None),
                            Point(x=1000, y=4000, z=None),
                            Point(x=100000, y=100000, z=None),
                        ]
                    ),
                ),
                timestamp_ms=1003,
            )
            tracklet.update(
                None,
                timestamp_ms=1005,
            )
            tracklet.update(
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
                timestamp_ms=1004,
            )
            print(tracklet.timestamps)
            self.assertTrue(
                list(tracklet.timestamps) == [1001, 1002, 1003, 1004]
            )
            self.assertTrue(tracklet.xysr_track.shape == (4, 4))
            print(tracklet.xysr_track)
            expected = np.array(
                [
                    [
                        1.00000000e1,
                        1.00000000e1,
                        5.00025000e4,
                        1.00000000e1,
                    ],
                    [
                        2.00000000e1,
                        2.00000000e1,
                        5.00075000e4,
                        2.00000000e1,
                    ],
                    [
                        1.00000000e2,
                        1.00000000e2,
                        9.99800008e9,
                        1.00000000e2,
                    ],
                    [
                        1.00000000,
                        1.00000000,
                        9.99899995e-1,
                        1.00000000,
                    ],
                ],
                dtype=np.float64,
            )
            self.assertTrue(
                np.allclose(
                    tracklet.xysr_track,
                    expected,
                )
            )

        def perf_test():
            tracklet = Tracklet()
            for time in range(10000):
                tracklet.update(
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
                    timestamp_ms=time,
                )

        # run perf test using timeit
        do_perf_test: bool = False
        if do_perf_test:
            print(timeit.timeit(perf_test, number=5))


if __name__ == "__main__":
    unittest.main()
