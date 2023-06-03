#
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

from typing import Optional, Union

from transitions import Machine

from core.incident_machine.machines.base import (
    BaseStateMachine,
    RelevantActorInfo,
)
from core.structs.actor import ActorCategory
from core.structs.event import Event, EventType
from core.structs.incident import Incident
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ActorMachine:
    # Maximum time range within which an actor is supposed to come to a stop
    # to be considered not in violation.
    MAX_TIME_WINDOW_TO_STOP_FOR_INTERSECTION_S = 5

    INCIDENT_THRESH_LOW = 0.4
    INCIDENT_THRESH_MED = 0.8
    INCIDENT_THRESH_HIGH = 1.1

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.camera_uuid = camera_uuid
        self.actor_info = actor_info

        if params is not None:
            self.INCIDENT_THRESH_LOW = params.get(
                "incident_thresh_low", self.INCIDENT_THRESH_LOW
            )
            self.INCIDENT_THRESH_MED = params.get(
                "incident_thresh_med", self.INCIDENT_THRESH_MED
            )
            self.INCIDENT_THRESH_HIGH = params.get(
                "incident_thresh_high", self.INCIDENT_THRESH_HIGH
            )
            self.MAX_TIME_WINDOW_TO_STOP_FOR_INTERSECTION_S = params.get(
                "max_time_window_to_stop_for_intersection_s",
                self.MAX_TIME_WINDOW_TO_STOP_FOR_INTERSECTION_S,
            )

        self.MAX_TIME_WINDOW_TO_STOP_FOR_INTERSECTION_MS = (
            1000 * self.MAX_TIME_WINDOW_TO_STOP_FOR_INTERSECTION_S
        )

        self.last_known_stop_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None
        self.velocity_magnitude_at_intersection_crossing = None

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        states = [
            "start",
            "ever_seen_stopped",
            "incident",
        ]

        transitions = [
            {
                "trigger": "update_pit_state",
                "source": ["start", "ever_seen_stopped"],
                "dest": "ever_seen_stopped",
                "conditions": "_is_pit_stationary",
            },
            {
                "trigger": "update_pit_state",
                "source": ["start", "ever_seen_stopped"],
                "dest": None,
            },
            {
                "trigger": "pit_intersection_crossed",
                "source": "start",
                "dest": "incident",
                "conditions": "_is_pit_going_too_fast_at_intersection",
            },
            {
                "trigger": "pit_intersection_crossed",
                "source": "ever_seen_stopped",
                "dest": "incident",
                "conditions": [
                    "_is_pit_going_too_fast_at_intersection",
                    "_is_stopped_outside_allowed_time_window",
                ],
            },
            {
                "trigger": "pit_intersection_crossed",
                "source": ["start", "ever_seen_stopped"],
                "dest": "start",
            },
            {
                "trigger": "try_reset_state",
                "source": "incident",
                "dest": "start",
            },
            {
                "trigger": "try_reset_state",
                "source": ["ever_seen_stopped", "start"],
                "dest": None,
            },
        ]
        Machine(
            model=self,
            states=states,
            transitions=transitions,
            initial="start",
            ignore_invalid_triggers=False,
        )

    def transition_state_event(
        self, state_event_message
    ) -> Optional[Incident]:

        incident_list = []

        self.state_event_start_timestamp_ms = state_event_message.timestamp_ms
        self.state_event_end_timestamp_ms = (
            state_event_message.end_timestamp_ms
        )

        if isinstance(state_event_message, State):
            self.update_pit_state(
                is_pit_stationary=state_event_message.pit_is_stationary
            )
        elif isinstance(state_event_message, Event):
            self.velocity_magnitude_at_intersection_crossing = (
                state_event_message.normalized_speed
                if state_event_message.normalized_speed is not None
                else 0.0
            )
            self.pit_intersection_crossed(
                velocity=self.velocity_magnitude_at_intersection_crossing,
                incident_list=incident_list,
            )

        self.try_reset_state()
        return incident_list[0] if len(incident_list) else None

    # Conditionals
    def _is_pit_going_too_fast_at_intersection(
        self, velocity, **kwargs
    ) -> bool:
        return velocity > self.INCIDENT_THRESH_LOW

    def _is_stopped_outside_allowed_time_window(self, **kwargs) -> bool:
        return (
            self.state_event_end_timestamp_ms
            - self.last_known_stop_timestamp_ms
            > self.MAX_TIME_WINDOW_TO_STOP_FOR_INTERSECTION_MS
        )

    def _is_pit_stationary(self, is_pit_stationary, **kwargs) -> bool:
        return is_pit_stationary

    # Callbacks
    def on_enter_ever_seen_stopped(self, **kwargs) -> None:
        self.last_known_stop_timestamp_ms = self.state_event_end_timestamp_ms

    def on_enter_start(self, **kwargs) -> None:
        self.last_known_stop_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None

    def on_enter_incident(self, incident_list, **kwargs):
        incident = Incident(
            camera_uuid=self.camera_uuid,
            start_frame_relative_ms=self.state_event_start_timestamp_ms,
            end_frame_relative_ms=self.state_event_end_timestamp_ms,
            actor_ids=[self.actor_info.actor_id],
            track_uuid=self.actor_info.track_uuid,
        )
        if (
            self.velocity_magnitude_at_intersection_crossing
            > self.INCIDENT_THRESH_HIGH
        ):
            incident.priority = "high"
        elif (
            self.velocity_magnitude_at_intersection_crossing
            > self.INCIDENT_THRESH_MED
        ):
            incident.priority = "medium"
        elif (
            self.velocity_magnitude_at_intersection_crossing
            > self.INCIDENT_THRESH_LOW
        ):
            incident.priority = "low"

        incident_list.append(incident)


class DoorIntersectionMachine(BaseStateMachine):

    NAME = "door_intersection"
    INCIDENT_TYPE_NAME = "No Stop at Door"
    INCIDENT_TYPE_ID = "NO_STOP_AT_DOOR_INTERSECTION"
    PRE_START_BUFFER_MS = 10000
    POST_END_BUFFER_MS = 10000
    INCIDENT_VERSION = "1.0"
    INCIDENT_PRIORITY = "low"

    # Cache Configs
    MAX_CACHE_SIZE = 20  # max 20 actors in a scene
    MAX_CACHE_TIME_S = 10 * 60  # 10 minute

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        if (
            isinstance(state_event_message, State)
            and state_event_message.actor_category == ActorCategory.PIT
        ):
            return RelevantActorInfo(
                actor_id=state_event_message.actor_id,
                track_uuid=state_event_message.track_uuid,
            )

        if (
            isinstance(state_event_message, Event)
            and state_event_message.event_type == EventType.PIT_EXITING_DOOR
        ):
            return RelevantActorInfo(
                actor_id=state_event_message.subject_id,
                track_uuid=state_event_message.subject_uuid,
            )

        return None

    def transition_actor_machine(
        self,
        actor_info: RelevantActorInfo,
        state_event_message: Union[State, Event],
    ) -> Optional[Incident]:

        if actor_info.track_uuid not in self.actor_machines:
            # create the machine first
            self.actor_machines[actor_info.track_uuid] = ActorMachine(
                self.camera_uuid,
                actor_info,
                params=self.params,
            )

        actor_machine = self.actor_machines[actor_info.track_uuid]
        return actor_machine.transition_state_event(state_event_message)
