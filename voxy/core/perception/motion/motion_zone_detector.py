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

import typing

import numpy as np
from loguru import logger

from core.perception.motion.motion_zone_detector_mog2 import (
    MotionZoneDetectorMOG2,
)
from core.structs.actor import Actor


class MotionZoneDetector:
    """
    Motion Detector Classifier
    """

    k_motion_detector_map = {
        "mog2": MotionZoneDetectorMOG2,
    }

    def __init__(
        self,
        motion_detector_type,
        camera_uuid,
        config,
    ):
        if motion_detector_type not in self.k_motion_detector_map:
            logger.error(f"{motion_detector_type} does not exist!")
            raise RuntimeError(
                f"Motion detector must be of type {self.k_motion_detector_map.keys()}"
            )

        self.classifier = self.k_motion_detector_map[motion_detector_type](
            camera_uuid, config=config
        )

    def __call__(self, frame: np.ndarray) -> typing.List[Actor]:
        """
        Args:
            frame (np.ndarray): input frame from perception node
        Returns:
            typing.List[Actor]: list of motion region actors
        """
        return self.classifier.detect_motion(frame)
