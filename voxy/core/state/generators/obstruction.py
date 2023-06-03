#
# Copyright 2023 Voxel Labs, Inc.
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

from core.state.generators.base import BaseStateGenerator
from core.structs.actor import ActorCategory
from core.structs.generator_response import GeneratorResponse
from core.structs.state import State
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class ObstructionStateGenerator(BaseStateGenerator):
    """
    Class that generates state and event messages for obstructions.
    """

    def __init__(self, config: dict) -> None:
        self._camera_uuid = config["camera_uuid"]
        self._config = config

    def _tracklet_stationary(
        self, tracklet: Tracklet, vignette: Vignette
    ) -> bool:
        return tracklet.is_stationary

    def _get_states(self, vignette: Vignette) -> typing.List:
        states = []

        for tid, tracklet in vignette.tracklets.items():
            if tracklet.category == ActorCategory.OBSTRUCTION:
                current_time = (
                    vignette.present_frame_struct.relative_timestamp_ms
                )
                current_actor = tracklet.get_actor_at_timestamp(current_time)
                if current_actor is None:
                    continue
                states.append(
                    State(
                        actor_id=str(tid),
                        track_uuid=current_actor.track_uuid,
                        actor_category=ActorCategory.OBSTRUCTION,
                        timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                        camera_uuid=self._camera_uuid,
                        obstruction_is_stationary=self._tracklet_stationary(
                            tracklet, vignette
                        ),
                        end_timestamp_ms=self._get_end_timestamp_ms(vignette),
                    )
                )

        return states

    def process_vignette(self, vignette: Vignette) -> GeneratorResponse:
        """
        Process States & Events

        Args:
            vignette: state of the world over a given period of time

        Returns:
            A GeneratorResponse object containing the generated states and events
        """
        states = self._get_states(vignette)
        return GeneratorResponse(events=[], states=states)
