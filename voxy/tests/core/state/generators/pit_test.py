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

from core.state.generators.pit import PitStateGenerator
from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Point, Polygon
from core.structs.frame import Frame
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class PitStateGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def get_polygon_from_cxywh(self, cx, cy, w, h):
        polygon_corners = [
            Point(cx - w / 2, cy - h / 2),
            Point(cx + w / 2, cy - h / 2),
            Point(cx + w / 2, cy + h / 2),
            Point(cx - w / 2, cy + h / 2),
            Point(cx - w / 2, cy - h / 2),
        ]
        return Polygon(polygon_corners)

    def test_is_in_driving_area(self) -> None:
        slight_overlap = self.get_polygon_from_cxywh(0, 0, 10, 10)
        large_overlap = self.get_polygon_from_cxywh(0, 0, 3, 3)
        fake_driving_region_poly = self.get_polygon_from_cxywh(0, 0, 3, 4)
        actor = Actor()
        fake_driving_region = Actor()
        fake_driving_region.polygon = fake_driving_region_poly
        actor.polygon = slight_overlap
        generator = PitStateGenerator({"camera_uuid": "/test/uuid"})
        self.assertEqual(None, generator.is_in_driving_area(None, None))
        # test for empty list
        self.assertEqual(None, generator.is_in_driving_area(actor, []))
        self.assertEqual(
            False, generator.is_in_driving_area(actor, [fake_driving_region])
        )
        actor.polygon = large_overlap
        self.assertEqual(
            True, generator.is_in_driving_area(actor, [fake_driving_region])
        )

    def test_get_drivable_areas(self):
        fake_driving_region = Actor()
        fake_driving_region.category = ActorCategory.DRIVING_AREA
        generator = PitStateGenerator({"camera_uuid": "/test/uuid"})
        vignette = Vignette()
        vignette.present_frame_struct = Frame(0, 1, 1, 0, 0, 0)
        vignette.present_frame_struct.actors = [fake_driving_region]
        result = generator.get_drivable_areas(vignette)
        self.assertEqual(len(result), 1)
        vignette.present_frame_struct.actors = []
        result = generator.get_drivable_areas(vignette)
        self.assertEqual(len(result), 0)

    def test_get_states(self):
        fake_driving_region = Actor()
        fake_driving_region.category = ActorCategory.DRIVING_AREA
        fake_driving_region_poly = self.get_polygon_from_cxywh(0, 0, 3, 4)
        fake_driving_region.polygon = fake_driving_region_poly
        fake_pit_actor = Actor()
        fake_pit_actor.category = ActorCategory.PIT
        fake_pit_poly = self.get_polygon_from_cxywh(0, 0, 3, 4)
        fake_pit_actor.polygon = fake_pit_poly

        generator = PitStateGenerator({"camera_uuid": "/test/uuid"})
        vignette = Vignette()
        vignette.present_frame_struct = Frame(0, 1, 1, 0, 0, 0)
        # add actor to tracklet
        driving_tracklet = Tracklet()
        driving_tracklet.update(fake_driving_region, 0)

        pit_tracklet = Tracklet()
        pit_tracklet.update(fake_pit_actor, 0)
        vignette.present_frame_struct.actors = [
            fake_driving_region,
            fake_pit_actor,
        ]
        vignette.present_timestamp_ms = 0
        vignette.tracklets = {100: driving_tracklet, 101: pit_tracklet}

        # test is stationary
        result = generator.get_drivable_areas(vignette)
        result = generator.process_vignette(vignette)
        self.assertTrue(result.states[0].pit_in_driving_area)
        self.assertTrue(not result.states[0].pit_is_stationary)
        pit_tracklet.is_stationary = True
        result = generator.process_vignette(vignette)
        self.assertTrue(result.states[0].pit_is_stationary)

    def test_generator_config(self):
        config = {
            "camera_uuid": "/test/uuid",
            "state": {"pit": {"drivable_region_threshold": -3.0}},
        }
        generator = PitStateGenerator(config)
        self.assertTrue(generator.DRIVABLE_REGION_OVERLAP_THRESHOLD == -3.0)


if __name__ == "__main__":
    unittest.main()
