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

import copy

import numpy as np
import scipy.ndimage

from core.perception.acausal.algorithms.base import BaseAcausalAlgorithm
from core.structs.actor import ActorCategory
from core.structs.attributes import RectangleXYXY
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


# The track smoothing algorithm is documented here:
# https://github.com/voxel-ai/voxel/wiki/Acausal
class TrackSmoothingAlgorithm(BaseAcausalAlgorithm):
    GAUSSIAN_SMOOTHING_SIGMA_MS = 2.0

    def __init__(self, config: dict) -> None:
        self._config = config

    def _smooth_tracklet(self, tracklet: Tracklet) -> None:
        tracklet = copy.deepcopy(tracklet)

        num_timestamps = len(tracklet.timestamps)
        if not tracklet or num_timestamps < 2:
            return tracklet

        # Get interpolated xysr trajectory
        xysr_states = np.array([])
        timestamps_smooth = np.linspace(
            tracklet.timestamps[0],
            tracklet.timestamps[-1],
            num_timestamps,
        )
        xysr_states = tracklet.get_xysr_at_time_range(timestamps_smooth)

        # Smooth trajectory with a gaussian filter and equal time intervals
        xysr_smooth = scipy.ndimage.gaussian_filter1d(
            xysr_states,
            axis=0,
            sigma=self.GAUSSIAN_SMOOTHING_SIGMA_MS,
            mode="nearest",
        ).transpose()

        # Set interpolation level
        interpolation_type = "linear"

        # Update tracklet with smoothed trajectory
        x_smooth_interp = scipy.interpolate.interp1d(
            timestamps_smooth,
            xysr_smooth[0, :],
            kind=interpolation_type,
            assume_sorted=True,
        )
        y_smooth_interp = scipy.interpolate.interp1d(
            timestamps_smooth,
            xysr_smooth[1, :],
            kind=interpolation_type,
            assume_sorted=True,
        )
        s_smooth_interp = scipy.interpolate.interp1d(
            timestamps_smooth,
            xysr_smooth[2, :],
            kind=interpolation_type,
            assume_sorted=True,
        )
        r_smooth_interp = scipy.interpolate.interp1d(
            timestamps_smooth,
            xysr_smooth[3, :],
            kind=interpolation_type,
            assume_sorted=True,
        )

        xysr_smooth_interp = np.vstack(
            [
                x_smooth_interp(tracklet.timestamps),
                y_smooth_interp(tracklet.timestamps),
                s_smooth_interp(tracklet.timestamps),
                r_smooth_interp(tracklet.timestamps),
            ]
        )

        # Update _instances as well
        tracklet.timestamps = np.array([])
        tracklet.xysr_track = np.empty((4, 0))
        for i, timestamp in enumerate(
            tracklet._track.keys()  # trunk-ignore(pylint/W0212)
        ):
            actor_ind = tracklet.get_closest_non_null_actor_at_time(timestamp)
            actor = copy.deepcopy(
                tracklet[actor_ind]
            )  # TODO(Vai): Check that the correct actor is indexed
            polygon = RectangleXYXY.from_list(
                # trunk-ignore(pylint/W0212)
                tracklet._convert_xysr_to_bbox(xysr_smooth_interp[:, i])
            ).to_polygon()
            actor.polygon = polygon
            tracklet.update(actor, timestamp)

        return tracklet

    def process_vignette(self, vignette: Vignette) -> None:
        vignette = copy.deepcopy(vignette)
        for i, tracklet in vignette.tracklets.items():
            if tracklet.category is ActorCategory.PIT:
                smoothed_tracklet = self._smooth_tracklet(tracklet)
                vignette.tracklets[i] = smoothed_tracklet
        return vignette
