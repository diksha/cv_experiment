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
from copy import deepcopy

from core.perception.acausal.algorithms.velocity_estimation import (
    VelocityEstimationAlgorithm,
)
from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Point, Polygon
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class VelocityEstimationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.algorithm = VelocityEstimationAlgorithm(
            {"camera_uuid": "uscold/laredo/dock01/cha"}
        )
        self.null_algorithm = VelocityEstimationAlgorithm(
            {"camera_uuid": "foo"}
        )
        self.dummy_tracklet = Tracklet()
        self.dummy_tracklet.category = ActorCategory.PIT
        self.tracklet_with_none = Tracklet()
        self.setup_tracklet()
        self.WORLD_VELOCITY_CHECK = False

        # add a few actor instances

    def get_polygon_from_cxywh(self, cx, cy, w, h):
        polygon_corners = [
            Point(cx - w / 2, cy - h / 2),
            Point(cx + w / 2, cy - h / 2),
            Point(cx + w / 2, cy + h / 2),
            Point(cx - w / 2, cy + h / 2),
            Point(cx - w / 2, cy - h / 2),
        ]
        return Polygon(polygon_corners)

    def setup_tracklet(self) -> None:
        actor = Actor(category=ActorCategory.PIT)
        cx = 100
        cy = 100
        w = h = 5
        # top left corner
        t = 0
        PIXEL_DISPLACEMENT = 2
        dt_ms = 10 * 1000  # time is in ms

        self.true_velocity = PIXEL_DISPLACEMENT / dt_ms * 1000
        for _ in range(10):
            cx += PIXEL_DISPLACEMENT
            cy += PIXEL_DISPLACEMENT
            print(cx)
            print(cy)
            new_polygon = self.get_polygon_from_cxywh(cx, cy, w, h)
            actor.polygon = new_polygon
            self.dummy_tracklet.update(deepcopy(actor), t)
            t += dt_ms

        new_actor = deepcopy(actor)
        new_actor.category = ActorCategory.PIT

        # only add one tracklet to check boundary conditions
        self.tracklet_with_none.update(deepcopy(new_actor), t)
        self.tracklet_with_none.update(deepcopy(new_actor), t + 1)

    def test_pixel_velocity_estimate(self) -> None:
        # test the pixel velocity estimate

        print(self.dummy_tracklet.category)
        self.dummy_tracklet.category = ActorCategory.PIT
        self.algorithm.update_pixel_velocity_estimate(
            self.dummy_tracklet, self.dummy_tracklet.timestamps[2]
        )
        print(self.dummy_tracklet.timestamps)
        print(self.dummy_tracklet.normalized_velocity_window)
        print(self.dummy_tracklet.x_velocity_pixel_per_sec)
        print(self.dummy_tracklet.y_velocity_pixel_per_sec)
        self.assertEqual(
            self.dummy_tracklet.x_velocity_pixel_per_sec, self.true_velocity
        )
        self.assertEqual(
            self.dummy_tracklet.y_velocity_pixel_per_sec, self.true_velocity
        )
        self.algorithm.update_pixel_velocity_estimate(self.dummy_tracklet, 0)

        self.assertEqual(self.dummy_tracklet.x_velocity_pixel_per_sec, 0)
        self.assertEqual(self.dummy_tracklet.y_velocity_pixel_per_sec, 0)

    def test_world_velocity_estimate(self) -> None:
        if not self.WORLD_VELOCITY_CHECK:
            return
        # TODO: enforce world velocity check, currently it is disabled because of runtime but this test should
        #      be updated

        self.algorithm.update_pixel_velocity_estimate(self.dummy_tracklet, 0)
        new_tracklet = self.dummy_tracklet
        self.algorithm.update_world_velocity_estimate(new_tracklet)
        # self.null_algorithm.update_world_velocity_estimate(new_tracklet)
        for timestamp, actor in new_tracklet.get_timestamps_and_actors():
            print(new_tracklet)
            print(actor.x_velocity_meters_per_sec)
            print(actor.y_velocity_meters_per_sec)
            if timestamp == 0:
                self.assertEqual(actor.x_velocity_meters_per_sec, None)
                self.assertEqual(actor.y_velocity_meters_per_sec, None)
            else:
                self.assertTrue(actor.x_velocity_meters_per_sec > 0)
                self.assertTrue(actor.y_velocity_meters_per_sec < 0)

    def test_velocity_with_one_actor(self) -> None:
        vignette = Vignette()
        vignette.tracklets = {1: self.tracklet_with_none}
        self.algorithm.process_vignette(vignette)
        new_tracklet = vignette.tracklets[1]
        for _, actor in new_tracklet.get_timestamps_and_actors():
            self.assertEqual(actor.x_velocity_meters_per_sec, None)
            self.assertEqual(actor.y_velocity_meters_per_sec, None)
