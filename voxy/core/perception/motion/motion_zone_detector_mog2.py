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

import cv2
import numpy as np
from loguru import logger

from core.common.functional.lib.compose import Compose
from core.common.functional.transforms.transforms import GaussianBlur
from core.perception.motion.motion_zone_detector_base import (
    MotionZoneDetectorBase,
)
from core.structs.actor import (
    Actor,
    ActorCategory,
    MotionDetectionZoneState,
    get_track_uuid,
)


class MotionZoneDetectorMOG2(MotionZoneDetectorBase):
    """
    MOG2 Motion Detection
    """

    k_default_motion_thresh = 0.008

    def __init__(self, camera_uuid: str, config: dict):
        super().__init__(camera_uuid)
        self.kernel_size = config.get("gaussian_kernel_size", 5)
        self._preprocessing = Compose(
            [GaussianBlur((self.kernel_size, self.kernel_size), 0)]
        )
        self.varthreshold = config.get("variance_threshold", 400)
        self.history = config.get("bg_frames_history", 500)
        self._fgbg = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.varthreshold,
            detectShadows=False,
        )
        if "motion_threshold" in config:
            logger.info("Loading custom motion threshold for config")
            self.k_default_motion_thresh = config["motion_threshold"].get(
                "default_motion_thresh", self.k_default_motion_thresh
            )

    def _generate_actors(self, motion_mask: np.ndarray) -> typing.List[Actor]:
        """
        Returns a list of motion detection zone actors
        Args:
            motion_mask (np.ndarray): fgmask output from mog2
        Returns:
            typing.List[Actor]: list of motion detection zone actors
        """
        actor_list = []
        for track_id, zone in enumerate(self._motion_detection_zones):
            # trunk-ignore(pylint/W0511)
            # TODO: resolve the actor ID issue
            actor_id = (
                track_id + 1000 * ActorCategory.MOTION_DETECTION_ZONE.value
            )
            zone_mask = self._get_motion_detection_zone_mask(zone.polygon)
            std_feature = motion_mask[zone_mask].std()
            motion_state = (
                MotionDetectionZoneState.MOTION
                if std_feature > self.k_default_motion_thresh
                else MotionDetectionZoneState.FROZEN
            )
            actor_list.append(
                Actor(
                    category=ActorCategory.MOTION_DETECTION_ZONE,
                    track_id=actor_id,
                    track_uuid=get_track_uuid(
                        camera_uuid=self._camera_uuid,
                        unique_identifier=str(zone.zone_id),
                        category=ActorCategory.MOTION_DETECTION_ZONE,
                    ),
                    polygon=zone.polygon,
                    manual=False,
                    motion_detection_zone_state=motion_state,
                    motion_detection_score_std=std_feature,
                )
            )
        return actor_list

    def _detect_motion(self, frame: np.ndarray) -> typing.List[Actor]:
        """
        Implementation of _detection_motion from MotionDetectionBase
        Args:
            frame (np.ndarray): input frame from percetion node
        Returns:
            typing.List[Actor]: List of motion detection zone actors
        """
        masked_frame = self._mask_frame(frame)
        preprocessed_frame = self._preprocessing(masked_frame)
        fgmask = self._fgbg.apply(preprocessed_frame)
        return self._generate_actors(fgmask)
