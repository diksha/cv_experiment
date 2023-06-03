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
from core.structs.event import Event
from core.structs.incident import Incident
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ActorMachine:
    # Threshold for how long a PIT can be parked/abandoned
    # before triggering a violation.
    MAX_PARKED_DURATION_S = 60  # TODO(twroge): restore this back to 120 seconds when the stationary fixes are in

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.actor_info = actor_info
        self.camera_uuid = camera_uuid

        if params is not None:
            self.MAX_PARKED_DURATION_S = params.get(
                "max_parked_duration_s", self.MAX_PARKED_DURATION_S
            )

        self.MAX_PARKED_DURATION_MS = self.MAX_PARKED_DURATION_S * 1000

        self.first_seen_in_parked_in_drivable_area_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        states = ["start", "parked_in_driving_area", "incident"]

        transitions = [
            {
                "trigger": "update_pit_state",
                "source": ["start"],
                "dest": "parked_in_driving_area",
                "conditions": [
                    "_is_stationary",
                    "_is_parked_in_driving_area",
                    "_is_not_associated",
                ],
            },
            # If we have already raised an incident then don't ever raise again for the same actor unless they
            # fail any of the condition which will put them back in start.
            {
                "trigger": "update_pit_state",
                "source": ["parked_in_driving_area", "incident"],
                "conditions": [
                    "_is_stationary",
                    "_is_parked_in_driving_area",
                    "_is_not_associated",
                ],
                "dest": None,
            },
            {
                "trigger": "update_pit_state",
                "source": ["start", "parked_in_driving_area", "incident"],
                "dest": "start",
            },
            {
                "trigger": "check_incident",
                "source": "parked_in_driving_area",
                "conditions": [
                    "_is_parked_for_greater_than_time_threshold",
                ],
                "dest": "incident",
                "after": "_get_incident",
            },
            {
                "trigger": "check_incident",
                "source": ["start", "parked_in_driving_area", "incident"],
                "dest": None,
            },
            {
                "trigger": "try_reset_state",
                "source": "incident",
                "dest": "start",
            },
            {
                "trigger": "try_reset_state",
                "source": [
                    "start",
                    "parked_in_driving_area",
                ],
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
        self, state_event_message: Union[State, Event]
    ) -> Optional[Incident]:

        incident_list = []
        self.state_event_start_timestamp_ms = state_event_message.timestamp_ms
        self.state_event_end_timestamp_ms = (
            state_event_message.end_timestamp_ms
        )

        self.update_pit_state(
            pit_is_stationary=state_event_message.pit_is_stationary,
            pit_in_driving_area=state_event_message.pit_in_driving_area,
            pit_is_associated=state_event_message.pit_is_associated,
        )

        self.check_incident(incident_list=incident_list)

        self.try_reset_state()

        return incident_list[0] if len(incident_list) > 0 else None

    def _is_parked_for_greater_than_time_threshold(self, **kwargs) -> bool:
        return (
            self.state_event_end_timestamp_ms
            - self.first_seen_in_parked_in_drivable_area_timestamp_ms
            > self.MAX_PARKED_DURATION_MS
        )

    def _is_stationary(self, pit_is_stationary, **kwargs) -> bool:
        return pit_is_stationary

    def _is_parked_in_driving_area(
        self, pit_in_driving_area, **kwargs
    ) -> bool:
        return pit_in_driving_area

    def _is_not_associated(self, pit_is_associated, **kwargs) -> bool:
        # TODO(harishma): We should be checking if pit_is_associated is False and
        # not allow zone transitions if pit_is_associated is None but currently our
        # positive scenario videos are too short which doesnt given association logic
        # enough time to switch from None to False.
        return not pit_is_associated

    # Callbacks
    def _get_incident(self, incident_list, **kwargs) -> None:
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.first_seen_in_parked_in_drivable_area_timestamp_ms,
                end_frame_relative_ms=self.state_event_end_timestamp_ms,
                actor_ids=[self.actor_info.actor_id],
                track_uuid=self.actor_info.track_uuid,
            )
        )

    def on_enter_parked_in_driving_area(self, **kwargs) -> None:
        self.first_seen_in_parked_in_drivable_area_timestamp_ms = (
            self.state_event_start_timestamp_ms
        )

    def on_enter_start(self, **kwargs) -> None:
        # reset everything
        self.first_seen_in_parked_in_drivable_area_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None


class ParkingMachine(BaseStateMachine):
    NAME = "parking"
    INCIDENT_TYPE_ID = "PARKING_DURATION"
    INCIDENT_TYPE_NAME = "Parking Duration"
    INCIDENT_VERSION = "1.3"
    INCIDENT_PRIORITY = "medium"

    PRE_START_BUFFER_MS = 10000
    POST_END_BUFFER_MS = 60000  # TODO(twroge): restore this back to something reasonable when the stationary fixes are in

    # Cache Configs
    MAX_CACHE_SIZE = 50  # max 50 actors in a scene
    MAX_CACHE_TIME_S = 10 * 60  # 10 minutes

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
