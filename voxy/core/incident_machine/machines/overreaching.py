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
from core.structs.ergonomics import PostureType
from core.structs.event import Event
from core.structs.incident import Incident
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ActorMachine:

    # Configs
    MAX_ALLOWED_BAD_REACH_DURATION_S = (
        0.5  # 500 ms evidence is enough to raise an incident
    )

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.camera_uuid = camera_uuid
        self.actor_info = actor_info

        if params is not None:
            self.MAX_ALLOWED_BAD_REACH_DURATION_S = params.get(
                "max_allowed_bad_reach_duration_s",
                self.MAX_ALLOWED_BAD_REACH_DURATION_S,
            )

        self.MAX_ALLOWED_BAD_REACH_DURATION_MS = (
            self.MAX_ALLOWED_BAD_REACH_DURATION_S * 1000
        )
        self.first_seen_in_bad_posture_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        states = ["start", "reaching_badly", "incident"]

        transitions = [
            {
                "trigger": "update_person_state",
                "source": "start",
                "dest": "reaching_badly",
                "conditions": [
                    "_is_reaching_badly",
                ],
            },
            # If we have already raised an incident then don't ever raise again for the same actor unless they
            # fail any of the condition which will put them back in start.
            {
                "trigger": "update_person_state",
                "source": ["reaching_badly", "incident"],
                "conditions": [
                    "_is_reaching_badly",
                ],
                "dest": None,
            },
            {
                "trigger": "update_person_state",
                "source": ["start", "reaching_badly", "incident"],
                "dest": "start",
            },
            {
                "trigger": "check_incident",
                "source": "reaching_badly",
                "conditions": [
                    "_is_reaching_badly_for_greater_than_time_threshold",
                ],
                "dest": "incident",
                "after": "_get_incident",
            },
            {
                "trigger": "check_incident",
                "source": ["start", "reaching_badly", "incident"],
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
                    "reaching_badly",
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

        self.update_person_state(
            person_reach_type=state_event_message.person_reach_type
        )

        self.check_incident(incident_list=incident_list)

        self.try_reset_state()

        return incident_list[0] if len(incident_list) > 0 else None

    def _is_reaching_badly_for_greater_than_time_threshold(
        self, **kwargs
    ) -> bool:
        return (
            self.state_event_end_timestamp_ms
            - self.first_seen_in_bad_posture_timestamp_ms
            > self.MAX_ALLOWED_BAD_REACH_DURATION_MS
        )

    def _is_reaching_badly(self, person_reach_type, **kwargs) -> bool:
        # Check only False and ignore None
        return person_reach_type == PostureType.BAD

    # Callbacks
    def _get_incident(self, incident_list, **kwargs) -> None:
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.first_seen_in_bad_posture_timestamp_ms,
                end_frame_relative_ms=self.state_event_end_timestamp_ms,
                actor_ids=[self.actor_info.actor_id],
                track_uuid=self.actor_info.track_uuid,
            )
        )

    def on_enter_reaching_badly(self, **kwargs) -> None:
        self.first_seen_in_bad_posture_timestamp_ms = (
            self.state_event_start_timestamp_ms
        )

    def on_enter_start(self, **kwargs) -> None:
        self.first_seen_in_bad_posture_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None


class OverReachingViolationMachine(BaseStateMachine):
    # State Machine Diagram:
    # https://drive.google.com/file/d/1MtoHAdIoXFi_PRxih-IEsf7-Far-joRS/view?usp=sharing

    NAME = "overreaching"
    INCIDENT_TYPE_ID = "OVERREACHING"
    INCIDENT_TYPE_NAME = "Over Reaching"
    PRE_START_BUFFER_MS = 10000
    POST_END_BUFFER_MS = 10000
    INCIDENT_VERSION = "1.2"
    INCIDENT_PRIORITY = "low"

    # Cache Configs
    MAX_CACHE_SIZE = 50  # max 50 actors in a scene
    MAX_CACHE_TIME_S = 10 * 60  # 10 minute

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        if (
            isinstance(state_event_message, State)
            and state_event_message.actor_category == ActorCategory.PERSON
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
