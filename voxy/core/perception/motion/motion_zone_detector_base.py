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
from abc import ABC, abstractmethod

import numpy as np
import rasterio.features
from loguru import logger

from core.incidents.utils import CameraConfig
from core.structs.actor import Actor
from core.structs.attributes import Polygon


class MotionZoneDetectorBase(ABC):
    """
    Base class for motion detection algorithms
    """

    def __init__(self, camera_uuid):
        self._motion_detection_zones = None
        self._camera_uuid = camera_uuid
        self._motion_detection_zones_mask = None
        self._contains_motion_detection_zones = (
            CameraConfig(camera_uuid, -1, -1).motion_detection_zones != []
        )
        if not self._contains_motion_detection_zones:
            logger.warning(
                f"Motion detection enabled for {camera_uuid} but no motion \
                detection zones defined, noop"
            )

    def _get_motion_detection_zone_mask(
        self, motion_detection_zone: Polygon
    ) -> np.ndarray:
        """
        Generates a bit mask representing the motion detection zone polygon
        with the size of _motion_detection_zones_mask
        Args:
            motion_detection_zone (Polygon): motion detection zone voxel polygon
        Returns:
            np.ndarray: bit mask of the motion detection zone
        Raises:
            RuntimeError: if the _motion_detection_zones_mask has not been initialized
        """
        if self._motion_detection_zones_mask is None:
            raise RuntimeError(
                "Tried calling _get_motion_detection_zone_mask on polygon \
                before we initialized the _motion_detection_zones_mask"
            )
        return rasterio.features.rasterize(
            [motion_detection_zone.to_shapely_polygon()],
            out_shape=self._motion_detection_zones_mask.shape,
        ).astype(bool)

    def _mask_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Masks frame with binary mask generated from motion detection zones in the camera config,
        if mask has not been initialized (called before we are able to get motion detection
        zones), retuns back original frame
        Args:
            frame (np.ndarray): frame to mask
        Returns:
            np.ndarray: input frame with _motion_detection_zones_mask applied to it
        Raises:
            RuntimeError: if frame has invalid dimensions
        """
        if self._motion_detection_zones_mask is None:
            logger.warning(
                "Cannot mask image, no motion detection zones available"
            )
            return frame

        if frame.ndim == 2:  # For grayscale images
            return frame * self._motion_detection_zones_mask
        if frame.ndim == 3:  # For images with color channels
            return frame * self._motion_detection_zones_mask[..., None]
        # Invalid frame dimensions
        raise RuntimeError(
            f"Cannot mask frame, input frame dimensions are invalid, ndim: {frame.ndim}"
        )

    def _initialize_motion_detection_zones_and_mask_if_required(
        self, frame: np.ndarray
    ) -> None:
        """
        Initialize the motion detection zone and bit mask when we get a
        frame to unnormalize camera config"
        Args:
            frame (np.ndarray): input frame from the perception node
        """
        if self._motion_detection_zones is None:
            self._motion_detection_zones = CameraConfig(
                self._camera_uuid, frame.shape[0], frame.shape[1]
            ).motion_detection_zones
            self._motion_detection_zones_mask = np.zeros(
                frame.shape[:2], dtype=bool
            )
            for motion_detection_zone_polygon in self._motion_detection_zones:
                self._motion_detection_zones_mask |= rasterio.features.rasterize(
                    [
                        motion_detection_zone_polygon.polygon.to_shapely_polygon()
                    ],
                    out_shape=self._motion_detection_zones_mask.shape,
                ).astype(
                    bool
                )

    @abstractmethod
    def _detect_motion(self, frame: np.ndarray) -> typing.List[Actor]:
        """
        Abstract method where motion detection algorithm is implemented by child class
        Args:
            frame (np.ndarray): input frame from perception node
        Returns:
            typing.List[Actor]: list of motion detection zone actors
        """
        raise NotImplementedError("_detect_motion not implemented")

    def detect_motion(self, frame: np.ndarray) -> typing.List[Actor]:
        """
        Wrapper for _detect_motion with check to verify that camera has motion
        detection zones, main function to call in MotionDetction class
        Args:
            frame (np.ndarray): input frame from perception node
        Returns:
            typing.List[Actor]: list of motion detection zone actors
        """
        if not self._contains_motion_detection_zones:
            return []
        self._initialize_motion_detection_zones_and_mask_if_required(frame)
        return self._detect_motion(frame)
