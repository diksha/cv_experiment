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
    # Threshold for how long an obstruction is stationary
    # before triggering a violation.
    MAX_OBSTRUCTION_DURATION_S = 60

    # Threshold for after how long to generate a new incident
    # for the same actor. To remove after getting out of experimental.
    # Directly affects number of cooldown incidents generated.
    MAX_INCIDENT_RESET_TIME_S = 600  # 10 mins

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.actor_info = actor_info
        self.camera_uuid = camera_uuid

        max_obstruction_duration_s = (
            params.get(
                "max_obstruction_duration_s", self.MAX_OBSTRUCTION_DURATION_S
            )
            if params is not None
            else self.MAX_OBSTRUCTION_DURATION_S
        )
        self.max_obstruction_duration_ms = max_obstruction_duration_s * 1000

        max_incident_reset_time_s = (
            params.get(
                "max_incident_reset_time_s", self.MAX_INCIDENT_RESET_TIME_S
            )
            if params is not None
            else self.MAX_INCIDENT_RESET_TIME_S
        )
        self.max_incident_reset_time_ms = max_incident_reset_time_s * 1000

        self.first_seen_in_area_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None
        self.last_obstruction_detected_timestamp_ms = None

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        states = ["start", "obstruction_in_area", "incident"]

        transitions = [
            {
                "trigger": "update_obstruction_state",
                "source": ["start"],
                "dest": "obstruction_in_area",
                "conditions": [
                    "_is_stationary",
                ],
            },
            # If we have already raised an incident then don't ever raise again
            # for the same actor unless they
            # fail any of the condition which will put them back in start.
            {
                "trigger": "update_obstruction_state",
                "source": ["obstruction_in_area", "incident"],
                "conditions": [
                    "_is_stationary",
                ],
                "dest": None,
            },
            {
                "trigger": "update_obstruction_state",
                "source": ["start", "obstruction_in_area", "incident"],
                "dest": "start",
            },
            {
                "trigger": "check_incident",
                "source": "obstruction_in_area",
                "conditions": [
                    "_is_in_area_for_greater_than_time_threshold",
                ],
                "dest": "incident",
                "after": "_get_incident",
            },
            {
                "trigger": "check_incident",
                "source": ["start", "obstruction_in_area", "incident"],
                "dest": None,
            },
            {
                "trigger": "try_reset_state",
                "source": "incident",
                "conditions": "_incident_reset_threshold_passed",
                "dest": "start",
            },
            {
                "trigger": "try_reset_state",
                "source": [
                    "start",
                    "obstruction_in_area",
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
        """Transition the state machine based on the state or event message.

        Args:
            state_event_message (Union[State, Event]): state or event message

        Returns:
            Optional[Incident]: incident if one was raised
        """

        incident_list = []
        self.state_event_start_timestamp_ms = state_event_message.timestamp_ms
        self.state_event_end_timestamp_ms = (
            state_event_message.end_timestamp_ms
        )

        self.update_obstruction_state(
            obstruction_is_stationary=state_event_message.obstruction_is_stationary,
        )

        self.check_incident(incident_list=incident_list)

        self.try_reset_state()

        return incident_list[0] if len(incident_list) > 0 else None

    # Conditionals
    def _incident_reset_threshold_passed(self, **kwargs: dict) -> bool:
        """
        Calculate whether a certain amount of time has passed since obstruction was last seen.
        If the threshold has passed then we can resume raising an incident when an obstruction.
        is detected.

        Returns:
            Whether a certain amount of time has passed since we entered incident state.
        """
        return (
            self.state_event_start_timestamp_ms
            - self.last_obstruction_detected_timestamp_ms
            > self.max_incident_reset_time_ms
        )

    def _is_in_area_for_greater_than_time_threshold(self, **kwargs) -> bool:
        return (
            self.state_event_end_timestamp_ms
            - self.first_seen_in_area_timestamp_ms
            > self.max_obstruction_duration_ms
        )

    def _is_stationary(self, obstruction_is_stationary, **kwargs) -> bool:
        return obstruction_is_stationary

    # Callbacks
    def _get_incident(self, incident_list, **kwargs) -> None:
        self.last_obstruction_detected_timestamp_ms = (
            self.state_event_end_timestamp_ms
        )
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.first_seen_in_area_timestamp_ms,
                end_frame_relative_ms=self.state_event_end_timestamp_ms,
                actor_ids=[self.actor_info.actor_id],
                track_uuid=self.actor_info.track_uuid,
            )
        )

    def on_enter_obstruction_in_area(self, **kwargs) -> None:
        self.first_seen_in_area_timestamp_ms = (
            self.state_event_start_timestamp_ms
        )

    def on_enter_start(self, **kwargs) -> None:
        # reset everything
        self.first_seen_in_area_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None


class ObstructionMachine(BaseStateMachine):
    NAME = "obstruction"
    INCIDENT_TYPE_ID = "OBSTRUCTION"
    INCIDENT_TYPE_NAME = "Obstruction"
    INCIDENT_VERSION = "experimental-1.3"
    INCIDENT_PRIORITY = "medium"

    PRE_START_BUFFER_MS = 10000
    POST_END_BUFFER_MS = 0

    # Default for after when to generate a new incident for the same actor.
    PER_ACTOR_COOLDOWN_THRESHOLD_S = 3600

    # Cache Configs
    MAX_CACHE_SIZE = 50  # max 50 actors in a scene
    MAX_CACHE_TIME_S = 10 * 60  # 10 minutes

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        """Get the relevant actor info from the state or event message.

        Args:
            state_event_message (Union[State, Event]): state or event message

        Returns:
            Optional[RelevantActorInfo]: relevant actor info
        """
        if (
            isinstance(state_event_message, State)
            and state_event_message.actor_category == ActorCategory.OBSTRUCTION
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
        """Transition the actor machine based on the state or event message.

        Args:
            actor_info (RelevantActorInfo): actor info
            state_event_message (Union[State, Event]): state or event message

        Returns:
            Optional[Incident]: incident if one was raised
        """

        if actor_info.track_uuid not in self.actor_machines:
            # create the machine first
            self.actor_machines[actor_info.track_uuid] = ActorMachine(
                self.camera_uuid,
                actor_info,
                params=self.params,
            )

        actor_machine = self.actor_machines[actor_info.track_uuid]
        return actor_machine.transition_state_event(state_event_message)
