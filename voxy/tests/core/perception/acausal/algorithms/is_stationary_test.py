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

from core.perception.acausal.algorithms.is_stationary import (
    IsStationaryAlgorithm,
)
from core.structs.actor import ActorCategory, ActorFactory
from core.structs.frame import Frame
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class IsStationaryAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.is_stationary_algorithm = IsStationaryAlgorithm(config={})
        self.present_timestamp_ms = 2000
        self.setup_tracklets()
        self.setup_vignette()

    def setup_vignette(self):
        """Sets up vignette for testing."""
        self.vignette = Vignette(
            present_timestamp_ms=self.present_timestamp_ms
        )
        self.vignette.tracklets = {
            self.pit_1_tracklet.track_id: self.pit_1_tracklet,
        }
        self.vignette.present_frame_struct = Frame(
            frame_number=0,
            frame_width=10,
            frame_height=10,
            relative_timestamp_s=0,
            relative_timestamp_ms=0,
            epoch_timestamp_ms=2000,
        )

    def setup_tracklets(self):
        """Sets up tracklets for testing."""
        self.pit_1_tracklet = Tracklet(
            track_id=1,
            category=ActorCategory.PIT,
            xysr_track=np.array([30, 20, 30, 1]).reshape(4, 1),
            normalized_velocity_window=np.array(
                [
                    0.1,
                    0.2,
                    0.3,
                ]
            ).reshape(3, 1),
        )
        self.pit_1_tracklet.update(
            ActorFactory.from_detection(
                "camera_uuid",
                1,
                [1, 2, 3, 4],
                ActorCategory.PIT,
                0.5,
            ),
            self.present_timestamp_ms - 1000,
        )
        self.pit_1_tracklet.update(
            ActorFactory.from_detection(
                "camera_uuid",
                1,
                [1, 2, 3, 4],
                ActorCategory.PIT,
                0.5,
            ),
            self.present_timestamp_ms,
        )
        self.pit_1_tracklet.update(
            ActorFactory.from_detection(
                "camera_uuid",
                1,
                [1, 2, 3, 4],
                ActorCategory.PIT,
                0.5,
            ),
            self.present_timestamp_ms + 1000,
        )

    def test_process_vignette_stationary(self):
        """Tests the process_vignette method."""
        vignette = self.is_stationary_algorithm.process_vignette(
            vignette=self.vignette
        )
        self.assertTrue(vignette.tracklets[1].is_stationary)

    def test_process_vignette_not_stationary(self):
        """Tests the process_vignette method."""
        self.pit_1_tracklet.normalized_velocity_window = np.array(
            [
                0,
                0.4,
                0.7,
            ]
        ).reshape(3, 1)
        vignette = self.is_stationary_algorithm.process_vignette(
            vignette=self.vignette
        )
        self.assertFalse(vignette.tracklets[1].is_stationary)
