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

import numpy as np
from skimage.filters import apply_hysteresis_threshold

from core.perception.acausal.algorithms.base import BaseAcausalAlgorithm
from core.structs.actor import ActorCategory
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


def compute_is_stationary(
    timestamps,
    PRE_WINDOW_MS,
    POST_WINDOW_MS,
    time_ms,
    normalized_velocity_window,
    HYSTERESIS_SPEED_LOW_NORMALIZED,
    HYSTERESIS_SPEED_HIGH_NORMALIZED,
):
    window_mask = np.logical_and(
        (timestamps > time_ms - PRE_WINDOW_MS),
        (timestamps < time_ms + POST_WINDOW_MS),
    )
    window_timestamps = timestamps[window_mask]
    speeds = normalized_velocity_window[window_mask]

    index = np.where(window_timestamps == time_ms)[0]

    ## apply hysteresis
    hysteresis_values = apply_hysteresis_threshold(
        speeds,
        low=HYSTERESIS_SPEED_LOW_NORMALIZED,
        high=HYSTERESIS_SPEED_HIGH_NORMALIZED,
    )
    # get the current timepoint actor
    above_threshold = hysteresis_values[index[0]]
    return not bool(above_threshold)


class IsStationaryAlgorithm(BaseAcausalAlgorithm):
    HYSTERESIS_SPEED_HIGH_NORMALIZED = 0.50
    HYSTERESIS_SPEED_LOW_NORMALIZED = 0.1
    PRE_WINDOW_MS = 4000
    POST_WINDOW_MS = 4000

    ACTORS_TO_UPDATE = [ActorCategory.PIT, ActorCategory.OBSTRUCTION]

    def __init__(self, config: dict) -> None:
        self._config = config

    def _update_is_stationary(self, tracklet: Tracklet, time_ms: int) -> None:
        if tracklet.category not in self.ACTORS_TO_UPDATE:
            return

        if len(tracklet) < 2:
            return

        if time_ms is None:
            tracklet.is_stationary = None
            return

        if time_ms not in tracklet.timestamps:
            tracklet.is_stationary = None
            return

        tracklet.is_stationary = compute_is_stationary(
            tracklet.timestamps,
            self.PRE_WINDOW_MS,
            self.POST_WINDOW_MS,
            time_ms,
            tracklet.normalized_velocity_window,
            self.HYSTERESIS_SPEED_LOW_NORMALIZED,
            self.HYSTERESIS_SPEED_HIGH_NORMALIZED,
        )
        return

    def process_vignette(self, vignette: Vignette) -> Vignette:
        for _, tracklet in vignette.tracklets.items():
            self._update_is_stationary(
                tracklet,
                vignette.present_frame_struct.epoch_timestamp_ms
                if vignette.present_frame_struct is not None
                else None,
            )
        return vignette
