#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.

# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import unittest

import numpy as np

from core.perception.acausal.algorithms.proximity.base_proximity_algorithm import (
    BaseProximityAlgorithm,
)

# test proximity controller
from core.perception.acausal.algorithms.proximity_controller import (
    ProximityAlgorithmController,
)
from core.structs.actor import ActorCategory
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class TestAlgorithm(BaseProximityAlgorithm):
    # trunk-ignore(pylint/W0246)
    def find_proximity(self, vignette):
        super().find_proximity(vignette)


class ProximityAlgorithmControllerTest(unittest.TestCase):
    def setUp(self):
        self.proximity_controller = ProximityAlgorithmController(config={})
        self.present_timestamp_ms = 1
        self.setup_tracklets()
        self.setup_vignette()
        self.setup_vignette_no_timestamp()
        self.setup_vignette_no_tracklets()
        self.setup_base_proximity_algorithm_controller()

    def setup_vignette(self):
        """Sets up vignette for testing."""
        self.vignette = Vignette(
            present_timestamp_ms=self.present_timestamp_ms
        )
        self.vignette.tracklets = {
            self.person_1_tracklet.track_id: self.person_1_tracklet,
            self.person_2_tracklet.track_id: self.person_2_tracklet,
            self.pit_1_tracklet.track_id: self.pit_1_tracklet,
            self.pit_2_tracklet.track_id: self.pit_2_tracklet,
        }

    def setup_tracklets(self):
        """Sets up tracklets for testing."""
        self.person_1_tracklet = Tracklet(
            track_id=1,
            category=ActorCategory.PERSON,
            is_associated_with_pit=True,
            xysr_track=np.array([30, 20, 30, 1]).reshape(4, 1),
            timestamps=np.array([self.present_timestamp_ms]),
        )
        self.person_2_tracklet = Tracklet(
            track_id=2,
            category=ActorCategory.PERSON,
            is_associated_with_pit=False,
            xysr_track=np.array([5, 20, 30, 1]).reshape(4, 1),
            timestamps=np.array([self.present_timestamp_ms]),
        )
        self.pit_1_tracklet = Tracklet(
            track_id=3,
            category=ActorCategory.PIT,
            xysr_track=np.array([6, 20, 70, 1]).reshape(4, 1),
            timestamps=np.array([self.present_timestamp_ms]),
        )
        self.pit_2_tracklet = Tracklet(
            track_id=4,
            category=ActorCategory.PIT,
            xysr_track=np.array([40, 20, 100, 1]).reshape(4, 1),
            timestamps=np.array([self.present_timestamp_ms]),
        )

    def setup_vignette_no_timestamp(self):
        self.vignette_timestamp = Vignette(present_timestamp_ms=None)

    def setup_vignette_no_tracklets(self):
        self.vignette_tracklets = Vignette(
            tracklets={}, present_timestamp_ms=1
        )

    def setup_base_proximity_algorithm_controller(self):
        """Test base proximity algorithm controller."""
        self.base_proximity_algorithm_controller = (
            ProximityAlgorithmController(config={})
        )

        # trunk-ignore(pylint/W0212)
        self.base_proximity_algorithm_controller._proximity_algorithms = [
            TestAlgorithm()
        ]

    def test_process_vignette(self):
        """Tests the process_vignette method."""
        vignette = self.proximity_controller.process_vignette(
            vignette=self.vignette
        )

        self.assertEqual(vignette.tracklets[3].is_proximal_to_person, True)
        self.assertEqual(vignette.tracklets[4].is_proximal_to_person, False)

        self.assertAlmostEqual(
            vignette.tracklets[3].nearest_person_pixel_proximity,
            float(1 / 100),
        )
        self.assertAlmostEqual(
            vignette.tracklets[4].nearest_person_pixel_proximity,
            float(35 / 130),
        )

        self.assertEqual(vignette.tracklets[3].is_proximal_to_pit, False)
        self.assertEqual(vignette.tracklets[4].is_proximal_to_pit, False)

        self.assertAlmostEqual(
            vignette.tracklets[3].nearest_pit_pixel_proximity,
            float(34 / 170),
        )

        # test vignette with no timestamp
        vignette = self.proximity_controller.process_vignette(
            self.vignette_timestamp
        )
        self.assertEqual(vignette, self.vignette_timestamp)

        # test vignette with no tracklets

        vignette = self.proximity_controller.process_vignette(
            self.vignette_tracklets
        )
        self.assertEqual(vignette, self.vignette_tracklets)

        # test base proximity algorithm controller
        with self.assertRaises(NotImplementedError):
            self.base_proximity_algorithm_controller.process_vignette(
                vignette=self.vignette
            )
