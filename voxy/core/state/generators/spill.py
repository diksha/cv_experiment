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

import typing

from core.state.generators.base import BaseStateGenerator
from core.structs.actor import ActorCategory
from core.structs.event import Event, EventType
from core.structs.generator_response import GeneratorResponse
from core.structs.vignette import Vignette


class SpillStateGenerator(BaseStateGenerator):
    """
    Class that generates state and event messages for spills.
    """

    def __init__(self, config: dict) -> None:
        self._camera_uuid = config["camera_uuid"]
        self._config = config

    def _get_events(self, vignette: Vignette) -> typing.List:
        """
        Generate any spill related events

        Args:
            vignette: state of the world over a given period of time

        Returns:
            A list of events related to spills
        """
        events = []
        spill_actor = self._get_spill_actor(vignette)
        if spill_actor is not None:
            events.append(
                Event(
                    timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                    camera_uuid=self._camera_uuid,
                    subject_id="spill_actor",
                    subject_uuid="spill_actor",
                    event_type=EventType.SPILL_DETECTED,
                    object_id=str(spill_actor.track_id),
                    object_uuid=str(spill_actor.track_uuid),
                    end_timestamp_ms=self._get_end_timestamp_ms(vignette),
                    x_velocity_pixel_per_sec=None,
                    y_velocity_pixel_per_sec=None,
                    normalized_speed=None,
                )
            )

        return events

    def _get_spill_actor(self, vignette: Vignette) -> int:
        """
        Gets the spill actor id from the vignette. The logic now is to
        simply return the largest

        TODO:
           consider if cumulative area over time is a better metric

        Args:
            vignette (Vignette): the vignette to find the spill actor in

        Returns:
            int: the largest spill actor id
        """
        spill_actors = []
        for _, tracklet in vignette.tracklets.items():
            if tracklet.category == ActorCategory.SPILL:
                current_time = (
                    vignette.present_frame_struct.relative_timestamp_ms
                )
                current_actor = tracklet.get_actor_at_timestamp(current_time)
                if current_actor is None:
                    continue
                spill_actors.append(current_actor)

        if not spill_actors:
            return None

        return max(spill_actors, key=lambda actor: actor.polygon.area())

    def process_vignette(self, vignette: Vignette) -> GeneratorResponse:
        """
        Process States & Events

        Args:
            vignette: state of the world over a given period of time

        Returns:
            A GeneratorResponse object containing the generated states and events
        """
        events = self._get_events(vignette)
        return GeneratorResponse(events=events, states=[])
