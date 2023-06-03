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

from loguru import logger

from core.perception.acausal.algorithms.base import BaseAcausalAlgorithm
from core.structs.actor import ActorCategory, MotionDetectionZoneState
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class MotionZoneSmoothing(BaseAcausalAlgorithm):
    """
    Applies temporal smoothing to motion detection zone state classification
    """

    k_smoothing_pre_window_ms = 4000
    k_smoothing_post_window_ms = 4000
    k_tracklet_motion_threshold = 0.20

    def __init__(self, config: dict):
        logger.info("Initializing motion_zone_smoothing")
        if "motion_detection" in config.get("incident", {}).get(
            "incident_machine_params", {}
        ):
            motion_detection_config = config["incident"][
                "incident_machine_params"
            ]["motion_detection"]
            self.k_smoothing_pre_window_ms = motion_detection_config.get(
                "smoothing_pre_window_ms", self.k_smoothing_pre_window_ms
            )
            self.k_smoothing_post_window_ms = motion_detection_config.get(
                "smoothing_post_window_ms", self.k_smoothing_post_window_ms
            )
            self.k_tracklet_motion_threshold = motion_detection_config.get(
                "tracklet_motion_threshold", self.k_tracklet_motion_threshold
            )

    def _smooth_motion_zone_state(
        self,
        motion_zone_state_list: list,
    ) -> MotionDetectionZoneState:
        """
        Checks to see if buffered motion zone states meet the required percentage required
        for being classified as motion
        Args:
            motion_zone_state_list (list): list of motion zone states
        Returns:
            smoothed motion detection zone state
        """
        count_motion_frames = motion_zone_state_list.count(
            MotionDetectionZoneState.MOTION
        )
        pct_motion_frames = float(count_motion_frames) / len(
            motion_zone_state_list
        )
        return pct_motion_frames > self.k_tracklet_motion_threshold

    def _update_motion_zone_state(
        self, tracklet: Tracklet, time_ms: int
    ) -> None:
        """
        Smooth out motion_detection_zone_state using pre and post buffers
        Args:
            tracklet (Tracklet): tracklet being processed
            time_ms (int): time of current frame in ms
        """
        if tracklet.category != ActorCategory.MOTION_DETECTION_ZONE:
            return

        if len(tracklet) < 2:
            return

        if time_ms is None:
            tracklet.is_motion_zone_in_motion = None
            return

        buffer_start = time_ms - self.k_smoothing_pre_window_ms
        buffer_end = time_ms + self.k_smoothing_post_window_ms
        # iterate through motion_zone_states between buffer_start and buffer_end
        motion_zone_state_list = [
            actor.motion_detection_zone_state
            for actor in list(
                tracklet.get_actor_instances_in_time_interval(
                    buffer_start, buffer_end
                )
            )
        ]

        if len(motion_zone_state_list) < 2:
            logger.warning(
                (",").join(
                    (
                        "No motion zones available to smooth in time range",
                        f"current_time,{time_ms}",
                        f"buffer_start,{buffer_start}",
                        f"buffer_end,{buffer_end}",
                        f"number_existing_instances,{len(tracklet)}",
                        f"first_instance_time,{tracklet.earliest_available_timestamp_ms()}",
                        f"last_instance_time,{tracklet.get_last_seen_timestamp()}",
                    )
                )
            )
            tracklet.is_motion_zone_in_motion = None
            return

        tracklet.is_motion_zone_in_motion = self._smooth_motion_zone_state(
            motion_zone_state_list
        )

    def process_vignette(self, vignette: Vignette) -> Vignette:
        """
        Process a vignette to smooth out motion detection zone tracklet status
        Args:
            vignette (Vignette): acaulal vignette
        Returns:
            Vignette: vignette with motion zone smooting applied
        """
        for _, tracklet in vignette.tracklets.items():
            self._update_motion_zone_state(
                tracklet,
                vignette.present_frame_struct.epoch_timestamp_ms
                if vignette.present_frame_struct is not None
                else None,
            )
        return vignette
