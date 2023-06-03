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
from core.structs.actor import Actor, ActorCategory, DoorOrientation, DoorState
from core.structs.event import Event, EventType
from core.structs.generator_response import GeneratorResponse
from core.structs.vignette import Vignette
from core.utils.struct_utils.utils import get_pct_intersection


class ActorStateGenerator(BaseStateGenerator):
    def __init__(self, config: dict) -> None:
        self._camera_uuid = config["camera_uuid"]
        # load configs
        self.config = config

    def _get_actor_crossing_events_for_front_door(
        self, door: Actor, vignette: Vignette, max_lookback: int
    ) -> list:
        """Generate Actor Entering Door or Actor Exiting Door events.

        When an actor crosses the door and enters the view of the camera we consider it an Entering Door
        event and when an actor crosses a door to move out of the view of the camera, we consider it an
        Exiting Door event.
        """
        events = []
        # If the door is not open, no crossing events can occur
        door_open = door.door_state in [
            DoorState.FULLY_OPEN,
            DoorState.PARTIALLY_OPEN,
        ]
        if not door_open:
            return events

        # trunk-ignore-all(pylint/R1702)
        for (
            track_id,
            tracklet,
        ) in vignette.tracklets.items():
            if tracklet.category not in [
                ActorCategory.PIT,
                ActorCategory.PERSON,
            ]:
                continue

            actor = tracklet.get_actor_at_timestamp(
                vignette.present_timestamp_ms
            )
            if actor is None:
                continue
            actor_track_id = track_id
            # First check if an actor is fully contained within the door. This indicates the
            # actor is currently on the side of the door away from the camera.
            actor_fully_contained_now = door.get_shapely_polygon().contains(
                actor.get_shapely_polygon()
            )
            stop_search = False
            # start from the most recent history and move backwards
            for index, historical_frame_struct in enumerate(
                vignette.past_frame_structs[::-1]
            ):
                if (max_lookback and index >= max_lookback) or stop_search:
                    break
                for historical_actor in historical_frame_struct.actors:
                    if historical_actor.track_id == actor_track_id:
                        actor_fully_contained_in_past = (
                            door.get_shapely_polygon().contains(
                                historical_actor.get_shapely_polygon()
                            )
                        )
                        if (
                            actor_fully_contained_in_past
                            != actor_fully_contained_now
                        ):
                            event_type = None
                            if actor.category == ActorCategory.PIT:
                                event_type = (
                                    EventType.PIT_EXITING_DOOR
                                    if actor_fully_contained_now
                                    else EventType.PIT_ENTERING_DOOR
                                )
                            elif actor.category == ActorCategory.PERSON:
                                event_type = (
                                    EventType.PERSON_EXITING_DOOR
                                    if actor_fully_contained_now
                                    else EventType.PERSON_ENTERING_DOOR
                                )
                            events.append(
                                Event(
                                    timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                                    camera_uuid=self._camera_uuid,
                                    subject_id=str(actor.track_id),
                                    subject_uuid=actor.track_uuid,
                                    event_type=event_type,
                                    object_id=str(door.track_id),
                                    object_uuid=door.track_uuid,
                                    end_timestamp_ms=self._get_end_timestamp_ms(
                                        vignette
                                    ),
                                    x_velocity_pixel_per_sec=tracklet.x_velocity_pixel_per_sec,
                                    y_velocity_pixel_per_sec=tracklet.x_velocity_pixel_per_sec,
                                    normalized_speed=tracklet.normalized_pixel_speed,
                                )
                            )
                        stop_search = True
        return events

    def _get_actor_crossing_events_for_side_door(
        self, door: Actor, vignette: Vignette, pct_threshold=0.9
    ) -> list:

        # We can only ever "see" actors exiting side doors
        events = []
        # If the door is not open, no crossing events can occur
        door_open = door.door_state in [
            DoorState.FULLY_OPEN,
            DoorState.PARTIALLY_OPEN,
        ]
        if not door_open:
            return events
        for (
            _,
            tracklet,
        ) in vignette.tracklets.items():
            if tracklet.category not in [
                ActorCategory.PIT,
                ActorCategory.PERSON,
            ]:
                continue

            actor = tracklet.get_actor_at_timestamp(
                vignette.present_timestamp_ms
            )
            if actor is None:
                continue

            # Check if pct_threshold of an actor is contained within the door.
            actor_pct_contained_now = (
                get_pct_intersection(actor.polygon, door.polygon)
                > pct_threshold
            )

            if actor_pct_contained_now:
                event_type = (
                    EventType.PIT_EXITING_DOOR
                    if actor.category == ActorCategory.PIT
                    else EventType.PERSON_EXITING_DOOR
                )
                events.append(
                    Event(
                        timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                        camera_uuid=self._camera_uuid,
                        subject_id=str(actor.track_id),
                        subject_uuid=actor.track_uuid,
                        event_type=event_type,
                        object_id=str(door.track_id),
                        object_uuid=door.track_uuid,
                        end_timestamp_ms=self._get_end_timestamp_ms(vignette),
                        x_velocity_pixel_per_sec=tracklet.x_velocity_pixel_per_sec,
                        y_velocity_pixel_per_sec=tracklet.x_velocity_pixel_per_sec,
                        normalized_speed=tracklet.normalized_pixel_speed,
                    )
                )

        return events

    def _get_events(self, vignette: Vignette) -> list:
        events = []
        for actor in vignette.present_frame_struct.actors:
            if (
                actor.door_orientation
                and actor.door_orientation == DoorOrientation.FRONT_DOOR
            ):
                events.extend(
                    self._get_actor_crossing_events_for_front_door(
                        actor, vignette, max_lookback=5
                    )
                )
            if (
                actor.door_orientation
                and actor.door_orientation == DoorOrientation.SIDE_DOOR
            ):
                pct_threshold = (self.config["state"]).get("IoU", 0.9)
                events.extend(
                    self._get_actor_crossing_events_for_side_door(
                        actor, vignette, pct_threshold
                    )
                )

        return events

    def _get_states(self, vignette) -> typing.List:
        states = []
        return states

    def process_vignette(self, vignette: Vignette) -> GeneratorResponse:
        states = self._get_states(vignette)
        events = self._get_events(vignette)
        return GeneratorResponse(events=events, states=states)
