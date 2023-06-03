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

import collections

from core.perception.temporal.buffer import Buffer
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class TemporalController:
    def __init__(self, config):
        self._config = config

        self._max_past_frames = self._config["temporal"]["max_past_frames"]
        self._max_future_frames = self._config["temporal"]["max_future_frames"]
        self._max_temporal_segmemnts = self._config["temporal"].get(
            "max_frame_segments",
            1,  # by default we hold no frames segments but the current
        )

        self._frame_struct_buffer = Buffer(
            max_past_frames=self._max_past_frames,
            max_future_frames=self._max_future_frames,
        )

        self._tracklets = collections.defaultdict(
            lambda: Tracklet(self._config["temporal"]["expire_threshold_ms"])
        )

        self._temporal_frame_segments = None

    def process_frame(self, current_frame_struct):
        (
            present_frame_struct,
            past_frame_structs,
            future_frame_structs,
        ) = self._frame_struct_buffer.process(current_frame_struct)

        past_frame_structs = [x for x in past_frame_structs if x is not None]
        future_frame_structs = [
            x for x in future_frame_structs if x is not None
        ]

        # Update tracklets
        tracklets_to_remove = []
        for tracklet_id, tracklet in self._tracklets.items():
            if tracklet.is_expired_at(
                current_frame_struct.relative_timestamp_ms
            ):
                tracklets_to_remove.append(tracklet_id)
            else:
                tracklet.update(
                    None, current_frame_struct.relative_timestamp_ms
                )

        for tracklet_id in tracklets_to_remove:
            self._tracklets.pop(tracklet_id)

        for actor in current_frame_struct.actors:
            tracklet = self._tracklets[actor.track_id]
            tracklet.update(actor, current_frame_struct.relative_timestamp_ms)

        return Vignette(
            tracklets=self._tracklets,
            past_frame_structs=past_frame_structs,
            future_frame_structs=future_frame_structs,
            present_frame_struct=present_frame_struct,
            present_timestamp_ms=present_frame_struct.relative_timestamp_ms
            if present_frame_struct is not None
            else None,
        )
