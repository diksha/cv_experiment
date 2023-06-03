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

from core.perception.acausal.algorithms.base import BaseAcausalAlgorithm
from core.structs.actor import ActorCategory
from core.structs.ergonomics import ActivityType, PostureType
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class ErgonomicsSmoothingAlgorithm(BaseAcausalAlgorithm):
    MIN_BAD_REACH = 3
    WINDOW_PRE_MS = 2000
    WINDOW_POST_MS = 2000

    def __init__(self, config):
        self._config = config
        self.camera_uuid = config["camera_uuid"]

    def _update_bad_reach_belief(
        self, tracklet: Tracklet, present_timestamp_ms: int
    ):
        """
        Updates the belief of if an actor is doing a bad reach. The main approach here
        is to take a window (5 seconds before and 5 seconds after) and if there have been
        more than MIN_BAD_REACH, then we update the belief.

        Args:
            tracklet (Tracklet): the tracklet for the present timestamp
            present_timestamp_ms (int): the current time
        """
        if present_timestamp_ms is None:
            return
        # get the list of actors from the tracklet
        instances = tracklet.get_actor_instances_in_time_interval(
            present_timestamp_ms - self.WINDOW_PRE_MS,
            present_timestamp_ms + self.WINDOW_POST_MS,
        )
        is_reaching_bad = [
            (
                instance.activity[ActivityType.REACHING.name].posture
                is PostureType.BAD
            )
            for instance in instances
            if instance is not None
            and instance.activity is not None
            and ActivityType.REACHING.name in instance.activity
        ]

        ## convert actors to bad lift belief
        # count the number of reaching bad
        bad_reaching_count = is_reaching_bad.count(True)
        current_actor = tracklet.get_actor_at_timestamp(present_timestamp_ms)
        if current_actor is None:
            return
        if bad_reaching_count >= self.MIN_BAD_REACH:
            tracklet.is_believed_to_be_in_unsafe_posture = True
        else:
            tracklet.is_believed_to_be_in_unsafe_posture = False

    def process_vignette(self, vignette: Vignette) -> Vignette:
        """
        Processes the vignette and updates the reach belief for
        PERSON actors

        Args:
            vignette (Vignette): the vignette to process

        Returns:
            Vignette: the transformed vignette
        """
        for _, tracklet in vignette.tracklets.items():
            if tracklet.category == ActorCategory.PERSON:
                self._update_bad_reach_belief(
                    tracklet, vignette.present_timestamp_ms
                )
        return vignette
