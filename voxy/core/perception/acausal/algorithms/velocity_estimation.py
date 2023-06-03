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

import os

import numpy as np
import yaml
from loguru import logger

from core.perception.acausal.algorithms.base import BaseAcausalAlgorithm
from core.perception.calibration.utils import (
    bounding_box_to_actor_position,
    calibration_config_to_camera_model,
)
from core.structs.actor import ActorCategory
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class VelocityEstimationAlgorithm(BaseAcausalAlgorithm):
    """VelocityEstimationAlgorithm.
    `
         The algorithm is documented here:
             https://docs.google.com/document/d/1PnEtOsMTa3BUhXrfdNZ7tUW7PBTguM0uilfv9DZxshw

        This module updates the world velocity and the pixel velocity of the tracklets

    """

    ACTORS_TO_UPDATE = [ActorCategory.PIT, ActorCategory.OBSTRUCTION]

    def __init__(self, config: dict) -> None:
        """__init__.

        Args:
            config (dict): the configuration for the run

        Returns:
            None:
        """
        self._config = config
        # load calibration config if it exists
        calibration_config = self.__get_calibration_config(self._config)
        self.camera_model = None
        if calibration_config is not None:
            self.camera_model = calibration_config_to_camera_model(
                calibration_config
            )
        self.dxdy_padding = np.array([[0], [0]])
        self.dt_padding = np.array([1])

    def __get_calibration_config(self, config: dict) -> dict:
        calibration_config = None
        # TODO: use a database query to grab this calibration
        if "camera_uuid" not in config:
            return None
        candidate_calibration_file = os.path.join(
            "configs/cameras/", config["camera_uuid"] + "_calibration.yaml"
        )
        if os.path.exists(candidate_calibration_file):
            with open(
                candidate_calibration_file,
                "r",
                encoding="utf8",
            ) as calibration_config_file:
                calibration_config = yaml.safe_load(calibration_config_file)
                logger.info("Loaded calibration config")
        else:
            logger.warning(
                f"Could not find calibration file @ {candidate_calibration_file} "
            )
        return calibration_config

    def update_pixel_velocity_estimate(
        self, tracklet: Tracklet, timestamp_ms: int
    ) -> None:
        """update_pixel_velocity_estimate.

        this updates the pixel velocity estimate using finite differences of the
        bounding box center

        Args:
            tracklet (tracklet): tracklet

        Returns:
            none:
        """

        if tracklet.category not in self.ACTORS_TO_UPDATE:
            return

        if timestamp_ms not in tracklet.timestamps:
            tracklet.normalized_pixel_speed = None
            tracklet.normalized_velocity_window = None
            return

        xy = tracklet.xysr_track[:2, :]
        s = tracklet.xysr_track[2, :]
        t = tracklet.timestamps
        dxdy = np.diff(xy, axis=1)
        dt = np.diff(t, axis=0) / 1000
        dxdy = np.hstack((self.dxdy_padding, dxdy))
        dt = np.hstack((self.dt_padding, dt))
        # some actors do not have their tracks updated
        if dxdy.shape[1] != dt.shape[0]:
            tracklet.normalized_pixel_speed = None
            tracklet.normalized_velocity_window = None
            return

        dxdy_dt = dxdy / dt
        normalized_dxdy_dt = np.linalg.norm(dxdy_dt / np.sqrt(s), axis=0)
        index = np.where(t == timestamp_ms)[0]

        if len(normalized_dxdy_dt) == 0:
            tracklet.normalized_pixel_speed = None
            tracklet.normalized_velocity_window = None
            return

        normalized_velocity = normalized_dxdy_dt[index[0]]
        tracklet.normalized_pixel_speed = normalized_velocity
        tracklet.x_velocity_pixel_per_sec = dxdy_dt[0][index[0]]
        tracklet.y_velocity_pixel_per_sec = dxdy_dt[1][index[0]]
        tracklet.normalized_velocity_window = normalized_dxdy_dt
        return

    def update_world_velocity_estimate(self, tracklet: Tracklet) -> None:
        """update_world_velocity_estimate.

        Args:
            tracklet (tracklet): tracklet

        Returns:
            none:
        """
        # TODO(twroge): see if the world estimate should be using the
        # raw pixel velocity estimate
        if self.camera_model is None:
            return
        # convert the 2D velocity to 3D
        for _, actor in tracklet.get_timestamps_and_actors():
            if (
                actor is not None
                and actor.x_velocity_pixel_per_sec is not None
                and actor.y_velocity_pixel_per_sec is not None
            ):
                vx_pps, vy_pps = (
                    actor.x_velocity_pixel_per_sec,
                    actor.y_velocity_pixel_per_sec,
                )
                u, v = bounding_box_to_actor_position(
                    actor.get_shapely_polygon()
                )
                vx_mps, vy_mps = self.camera_model.get_world_velocity(
                    u, v, vx_pps, vy_pps
                )
                actor.x_velocity_meters_per_sec = vx_mps
                actor.y_velocity_meters_per_sec = vy_mps

    def update_velocity_estimate(
        self, tracklet: Tracklet, timestamp_ms: int
    ) -> None:
        """update_velocity_estimate.

        Args:
            tracklet (Tracklet): tracklet

        Returns:
            None:
        """
        self.update_pixel_velocity_estimate(tracklet, timestamp_ms)
        return tracklet

    def process_vignette(self, vignette: Vignette) -> None:
        """process_vignette.

        Args:
            vignette (Vignette): vignette

        Returns:
            None:
        """
        for _, tracklet in vignette.tracklets.items():
            if vignette.present_frame_struct is not None:
                self.update_velocity_estimate(
                    tracklet,
                    vignette.present_frame_struct.relative_timestamp_ms,
                )
        return vignette
