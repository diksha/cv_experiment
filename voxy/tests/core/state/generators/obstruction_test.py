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

from core.state.generators.obstruction import ObstructionStateGenerator
from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Point, Polygon
from core.structs.frame import Frame
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class ObstructionStateGeneratorTest(unittest.TestCase):
    def get_polygon_from_cxywh(
        self, center_x, center_y, width, height
    ) -> Polygon:
        """Get polygon from center x, center y, width, height

        Args:
            center_x (int): center x
            center_y (int): center y
            width (int): width of polygon
            height (int): height of polygon

        Returns:
            Polygon: Polygon for points
        """
        polygon_corners = [
            Point(center_x - width / 2, center_y - height / 2),
            Point(center_x + width / 2, center_y - height / 2),
            Point(center_x + width / 2, center_y + height / 2),
            Point(center_x - width / 2, center_y + height / 2),
            Point(center_x - width / 2, center_y - height / 2),
        ]
        return Polygon(polygon_corners)

    def test_get_states(self):
        fake_obstruction_actor = Actor()
        fake_obstruction_actor.category = ActorCategory.OBSTRUCTION
        fake_obstruction_poly = self.get_polygon_from_cxywh(0, 0, 3, 4)
        fake_obstruction_actor.polygon = fake_obstruction_poly

        generator = ObstructionStateGenerator({"camera_uuid": "/test/uuid"})
        vignette = Vignette()
        vignette.present_frame_struct = Frame(0, 1, 1, 0, 0, 0)
        # add actor to tracklet
        obstruction_tracklet = Tracklet()
        obstruction_tracklet.update(fake_obstruction_actor, 0)
        vignette.present_frame_struct.actors = [
            fake_obstruction_actor,
        ]
        vignette.present_timestamp_ms = 0
        vignette.tracklets = {100: obstruction_tracklet}

        # test is stationary
        result = generator.process_vignette(vignette)
        self.assertFalse(result.states[0].obstruction_is_stationary)
        obstruction_tracklet.is_stationary = True
        result = generator.process_vignette(vignette)
        self.assertTrue(result.states[0].obstruction_is_stationary)
