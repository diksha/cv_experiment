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
from core.structs.actor import DoorState as DoorStateEnum
from core.structs.event import Event, EventType
from core.structs.generator_response import GeneratorResponse
from core.structs.state import State


class DoorStateGenerator(BaseStateGenerator):
    def __init__(self, config: dict):
        self._camera_uuid = config["camera_uuid"]

    def _get_states(self, vignette) -> typing.List[State]:
        states = []
        for actor in vignette.present_frame_struct.actors:
            if actor.category == ActorCategory.DOOR:
                is_open = (
                    actor.door_state == DoorStateEnum.FULLY_OPEN
                    or actor.door_state == DoorStateEnum.PARTIALLY_OPEN
                )

                states.append(
                    State(
                        actor_id=str(actor.track_id),
                        track_uuid=actor.track_uuid,
                        actor_category=ActorCategory.DOOR,
                        timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                        camera_uuid=self._camera_uuid,
                        door_is_open=is_open,
                        door_state=actor.door_state,
                        end_timestamp_ms=self._get_end_timestamp_ms(vignette),
                    )
                )
        return states

    def _get_events(self, vignette):
        events = []
        previous_frame_doors_state = {}

        if len(vignette.past_frame_structs):
            previous_frame_struct = vignette.past_frame_structs[-1]
            for actor in previous_frame_struct.actors:
                if actor.category == ActorCategory.DOOR:
                    previous_frame_doors_state[
                        actor.track_id
                    ] = actor.door_state

        for actor in vignette.present_frame_struct.actors:
            if actor.category == ActorCategory.DOOR:
                prev_state = previous_frame_doors_state.get(actor.track_id)
                if prev_state is not None and actor.door_state != prev_state:
                    event_type = (
                        EventType.DOOR_OPENED
                        if actor.door_state == DoorStateEnum.FULLY_OPEN
                        else EventType.DOOR_CLOSED
                        if actor.door_state == DoorStateEnum.FULLY_CLOSED
                        else EventType.DOOR_PARTIALLY_OPENED
                    )
                    events.append(
                        Event(
                            timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                            camera_uuid=self._camera_uuid,
                            subject_id=None,
                            subject_uuid=None,
                            event_type=event_type,
                            object_id=str(actor.track_id),
                            object_uuid=actor.track_uuid,
                            end_timestamp_ms=self._get_end_timestamp_ms(
                                vignette
                            ),
                        )
                    )
        return events

    def process_vignette(self, vignette) -> GeneratorResponse:
        states = self._get_states(vignette)
        events = self._get_events(vignette)
        return GeneratorResponse(events=events, states=states)
